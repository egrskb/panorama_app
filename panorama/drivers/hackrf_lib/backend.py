"""
Refactored HackRF backend with improved multi‑device support, peak detection
and full sweep assembly.  This implementation merges the useful pieces of the
original and proposed versions: it supports both single‑SDR and multi‑SDR
modes, exposes tunable detector parameters, and reuses the sweep assembler
to rebuild complete sweeps for downstream consumers (e.g. autopic tables).

Key improvements and behaviours:

* Unified entrypoint: the backend automatically uses multi‑SDR when more
  than one device is requested.  A single worker thread handles data
  acquisition to minimise contention inside the C library (FFTW is not
  thread‑safe across multiple Python threads).
* Safe library loading: the CFFI interface is declared once and then the
  library is loaded from a set of candidate names.  If loading fails an
  explicit RuntimeError is raised.
* Device enumeration: `list_serials()` returns the list of attached HackRF
  device serials without triggering unnecessary heavy initialisation of
  the hackrf driver.  Multi‑SDR enumeration is followed by `hq_close_all()`
  to ensure devices do not start streaming inadvertently.
* Tunable detector: `set_detector_params()` accepts threshold, minimum
  width, minimum sweeps and timeout and forwards these into the C library.
* Sweep assembly: both single and multi workers use `_SweepAssembler` to
  combine partial spectrum segments into a full sweep.  The assembled
  sweep is emitted via `fullSweepReady` to allow the UI to update peak
  tables and waterfalls.  Autopic tables rely on these assembled sweeps.
* Peak handling and watchlist: in multi‑SDR mode the worker fetches
  recent peaks and only queries the watchlist when peaks are present.
  This prevents flooding the queue with stale entries and reduces CPU
  overhead.  Wideband signals are forwarded as `SweepLine` instances via
  `sweepLine.emit`.

This file replaces previous iterations of `backend.py`.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from PyQt5 import QtCore
from cffi import FFI

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.shared.parsing import SweepLine


def _find_library() -> List[str]:
    """Return a list of candidate library names for dlopen.

    The search order is:
    1. In the same directory as this file.
    2. In the project root (two levels up).
    3. Plain names (to search system paths).
    """
    here = os.path.abspath(os.path.dirname(__file__))
    names = ["libhackrf_multi.so", "libhackrf_multi.dylib", "hackrf_multi.dll"]
    old = ["libhackrf_qsa.so", "libhackrf_qsa.dylib", "hackrf_qsa.dll"]
    candidates: List[str] = []
    for n in names + old:
        candidates.append(os.path.join(here, n))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    for n in names + old:
        candidates.append(os.path.join(root, n))
    candidates.extend(names + old)
    return candidates


class HackRFLibSource(SourceBackend):
    """HackRF backend using CFFI with optional multi‑SDR support.

    Depending on the number of requested devices (`set_num_devices`), the
    backend operates either in single‑SDR or multi‑SDR mode.  A single worker
    thread is used to avoid concurrent calls into the C library which relies
    on FFTW.  Detector parameters can be tuned at runtime via
    `set_detector_params`.  Full sweeps are assembled and emitted to the
    frontend via the `fullSweepReady` signal.
    """

    def __init__(self, lib_path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._ffi = FFI()
        # Define the C interface; these definitions apply to both single and
        # multi‑SDR modes.  Functions unused in single mode are simply never
        # invoked.
        self._setup_cdefs()
        self._lib = self._load_library(lib_path)
        # Worker thread (single or multi).
        self._worker: Optional[_BaseWorker] = None
        # Sweep assembler shared across workers.
        self._assembler = _SweepAssembler()
        # Number of devices requested.  Default to one (single mode).
        self._num_devices = 1
        # Detector parameters (threshold offset in dB, minimum width bins,
        # minimum sweeps, timeout in seconds).
        self._detector_params: Dict[str, float | int] = {
            'threshold_offset': 20.0,
            'min_width': 3,
            'min_sweeps': 3,
            'timeout': 2.0,
        }
        # Running flag used to track whether a worker is active.
        self._running = False
        # Cache for device enumeration to avoid repeated heavy queries
        self._serials_cache: Tuple[List[str], float] = ([], 0.0)

    # ------------------------------------------------------------------
    # CFFI binding helpers

    def _setup_cdefs(self) -> None:
        """Define the C API signatures used by this backend.

        When multiple instances of this backend are created within the same
        process, CFFI may warn about duplicate declarations.  To avoid
        warnings, we explicitly allow overriding existing declarations.
        """
        # The `override=True` argument prevents warnings about duplicate
        # declarations when the backend is instantiated multiple times.
        self._ffi.cdef(
            r"""
            typedef unsigned int uint32_t;
            typedef unsigned long long uint64_t;
            typedef struct {
                double f_center_hz;
                double bw_hz;
                float rssi_ema;
                uint64_t last_ns;
                int hit_count;
            } WatchItem;
            typedef struct {
                double f_hz;
                float rssi_dbm;
                uint64_t last_ns;
            } Peak;
            typedef struct {
                int master_running;
                int slave_running[2];
                double retune_ms_avg;
                int watch_items;
            } HqStatus;
            int  hq_open_all(int num_expected);
            void hq_close_all(void);
            int  hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz);
            int  hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on);
            int  hq_config_set_freq_range(double start_hz, double stop_hz, double step_hz);
            int  hq_config_set_dwell_time(uint32_t dwell_ms);
            void hq_set_detector_params(float threshold_offset_db, int min_width_bins,
                                       int min_sweeps, float timeout_sec);
            int  hq_start(void);
            void hq_stop(void);
            int  hq_get_watchlist_snapshot(WatchItem* out, int max_items);
            int  hq_get_recent_peaks(Peak* out, int max_items);
            int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
            void hq_set_grouping_tolerance_hz(double delta_hz);
            void hq_set_ema_alpha(float alpha);
            int  hq_get_status(HqStatus* out);
            int  hq_list_devices(char* serials[], int max_count);
            int  hq_get_device_count(void);
            // Legacy single‑SDR API declarations removed to avoid conflicting
            // definitions of hq_start()/hq_stop() that accept callback
            // parameters.  The multi‑SDR API uses hq_start(void) and
            // hq_stop(void) instead.
        """,
            override=True
        )

    def _load_library(self, lib_path: Optional[str]) -> Any:
        """Attempt to load the HackRF library from a list of candidate paths."""
        paths = [lib_path] if lib_path else _find_library()
        for p in paths:
            if not p:
                continue
            try:
                lib = self._ffi.dlopen(p)
                print(f"Loaded library: {p}")
                return lib
            except Exception:
                continue
        raise RuntimeError("Не удалось загрузить библиотеку HackRF")

    # ------------------------------------------------------------------
    # Public API methods

    def set_num_devices(self, num: int) -> None:
        """Specify the number of devices to use (1 for single, >1 for multi).

        When changing the number of devices while the backend is running, the
        current worker will be stopped and restarted.  The number of devices
        greater than 1 triggers multi‑SDR mode.  In multi mode, the library
        will attempt to use exactly `num` devices; if fewer are available,
        `start()` will emit an error.
        """
        if num < 1 or num > 3:
            raise ValueError("Number of devices must be between 1 and 3")
        # Restart if running with a different configuration
        restart = self._running and (self._num_devices != num)
        self._num_devices = num
        if restart:
            self.stop()
            # small delay to allow devices to settle
            time.sleep(0.1)
        mode = "Multi-SDR" if self._num_devices > 1 else "Single-SDR"
        print(f"📡 Режим: {mode} ({self._num_devices} устройств)")

    def set_detector_params(self, threshold_offset: float, min_width: int,
                            min_sweeps: int, timeout: float) -> None:
        """Update detector parameters for peak grouping.

        These values are forwarded to the C library whenever the worker is
        running.  They can safely be changed on the fly.
        """
        self._detector_params = {
            'threshold_offset': threshold_offset,
            'min_width': min_width,
            'min_sweeps': min_sweeps,
            'timeout': timeout,
        }
        # Apply immediately if running
        if self._running and self._lib:
            try:
                self._lib.hq_set_detector_params(
                    float(threshold_offset), int(min_width), int(min_sweeps), float(timeout)
                )
                print(
                    f"Detector params updated: +{threshold_offset:.1f} dB, width {min_width} bins, "
                    f"{min_sweeps} sweeps, timeout {timeout:.1f}s"
                )
            except Exception as e:
                print(f"Failed to set detector params: {e}")

    def list_serials(self) -> List[str]:
        """Return a list of detected HackRF serial numbers.

        Device enumeration is attempted via the multi‑SDR API.  If that fails
        (e.g. multi‑SDR library not loaded), the single‑SDR API is used as a
        fallback.  As some operating systems require explicit initialisation
        of the HackRF driver before devices are visible, this method will
        try to initialise the driver via libhackrf before querying the list.
        The driver is shut down immediately afterwards to avoid resource
        leakage.
        """
        # Serve from cache if fresh
        now = time.time()
        cached, ts = self._serials_cache
        if cached and (now - ts) < 2.0:
            return list(cached)
        out: List[str] = []
        if self._lib is None:
            return out
        # Optionally initialise the HackRF driver to populate device list
        hackrf = None
        try:
            import ctypes
            for name in ("libhackrf.so.0", "libhackrf.dll", "libhackrf.dylib"):
                try:
                    hackrf = ctypes.CDLL(name)
                    break
                except Exception:
                    continue
            if hackrf is not None and hasattr(hackrf, "hackrf_init"):
                try:
                    if hackrf.hackrf_init() != 0:
                        hackrf = None
                except Exception:
                    hackrf = None
        except Exception:
            hackrf = None
        # Try multi‑SDR API first.  Some versions of the multi‑SDR library
        # implicitly open devices when listing them, so we explicitly close
        # all devices after enumeration to avoid automatic streaming.
        try:
            count = self._lib.hq_get_device_count()
            if count > 0:
                serials = self._ffi.new("char*[]", count)
                for i in range(count):
                    serials[i] = self._ffi.new("char[]", 128)
                actual = self._lib.hq_list_devices(serials, count)
                for i in range(actual):
                    if serials[i] != self._ffi.NULL:
                        s = self._ffi.string(serials[i]).decode(errors="ignore")
                        if s and s != "0000000000000000":
                            out.append(s)
                # Close any devices that were implicitly opened during enumeration
                try:
                    self._lib.hq_close_all()
                except Exception:
                    pass
        except Exception:
            pass
        # Note: avoid calling legacy single-SDR APIs which are not declared in cdef
        # Shut down HackRF driver if we initialised it
        try:
            if hackrf is not None and hasattr(hackrf, "hackrf_exit"):
                hackrf.hackrf_exit()
        except Exception:
            pass
        # Update cache
        self._serials_cache = (list(out), now)
        return list(out)

    def is_running(self) -> bool:
        """Check if the worker thread is active."""
        return self._running and self._worker is not None and self._worker.is_alive()

    def start(self, config: SweepConfig) -> None:
        """Start data acquisition with the given sweep configuration.

        If the backend is already running, `start()` returns immediately with
        a status message.  The worker type is chosen based on the number of
        requested devices: one device for single mode, multiple devices for
        multi mode.  Device availability is checked before starting; if
        insufficient hardware is present, an error signal is emitted.
        """
        if self.is_running():
            self.status.emit("Уже запущено")
            return
        available = self.list_serials()
        required = self._num_devices if self._num_devices > 1 else 1
        print(f"🚀 Запуск в режиме: {'Multi-SDR' if required > 1 else 'Single-SDR'}")
        print(f"📡 Найдено устройств: {len(available)}")
        if len(available) < required:
            self.error.emit(f"Требуется {required} устройств, найдено {len(available)}")
            return
        # Configure the assembler for the requested sweep
        self._assembler.configure(config)
        # Select and start the worker
        if required > 1:
            self._worker = _MultiWorker(self, self._ffi, self._lib, config, required, self._assembler)
        else:
            self._worker = _SingleWorker(self, self._ffi, self._lib, config, self._assembler)
        self._worker.finished_sig.connect(self._on_worker_finished)
        self._worker.start()
        self._running = True
        self.started.emit()
        self.status.emit(f"Запущен {'Multi' if required > 1 else 'Single'}-SDR")

    def stop(self) -> None:
        """Stop the current worker and release devices."""
        if not self.is_running():
            return
        print("⏹️ Остановка SDR...")
        self.status.emit("Остановка...")
        if self._worker:
            self._worker.stop()
            # Wait for thread to exit gracefully
            self._worker.join(timeout=2.0)
            self._worker = None
        self._running = False
        self.finished.emit(0)
        self.status.emit("Остановлено")

    def _on_worker_finished(self, code: int, msg: str) -> None:
        """Handle worker completion.  Reset running state and emit signals."""
        if code != 0:
            self.error.emit(msg or f"Завершился с кодом {code}")
        self._worker = None
        self._running = False
        self.finished.emit(code)

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Query the current status from the C library (multi mode only).

        Returns a dict with keys `master_running`, `slave1_running`,
        `slave2_running`, `retune_ms` and `watch_items` if the backend is
        active in multi mode; otherwise returns None.
        """
        if not self.is_running() or self._num_devices <= 1:
            return None
        try:
            status = self._ffi.new("HqStatus*")
            if self._lib.hq_get_status(status) == 0:
                s = status[0]
                return {
                    'master_running': bool(s.master_running),
                    'slave1_running': bool(s.slave_running[0]),
                    'slave2_running': bool(s.slave_running[1]),
                    'retune_ms': s.retune_ms_avg,
                    'watch_items': s.watch_items,
                }
        except Exception:
            pass
        return None


class _BaseWorker(QtCore.QObject, threading.Thread):
    """Base class for worker threads handling HackRF data acquisition.

    Subclasses should implement the `run()` method to perform device
    initialization, configuration, data reading and cleanup.  This base
    class provides the common signal and stopping mechanism.
    """
    finished_sig = QtCore.pyqtSignal(int, str)

    def __init__(self) -> None:
        QtCore.QObject.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        self._running = True
        self._stop_ev = threading.Event()

    def stop(self) -> None:
        """Request the worker to stop."""
        self._running = False
        self._stop_ev.set()


class _SingleWorker(_BaseWorker):
    """Worker for single-SDR mode using the multi-SDR API.

    This worker opens exactly one device with `hq_open_all(1)`, configures
    sampling parameters, frequency range and dwell time, applies detector
    settings and then continuously reads the master spectrum via
    `hq_get_master_spectrum`.  Partial sweeps are assembled into full
    sweeps using `_SweepAssembler`.  Each assembled sweep is emitted via
    `fullSweepReady`.
    """

    def __init__(self, parent: HackRFLibSource, ffi: FFI, lib: Any, config: SweepConfig,
                 assembler: _SweepAssembler) -> None:
        super().__init__()
        self._parent = parent
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._assembler = assembler
        # counters for logging
        self._spectrum_updates = 0

    def run(self) -> None:
        code = 0
        msg = ""
        try:
            # Open a single device via multi-SDR API
            if self._lib.hq_open_all(1) != 0:
                self.finished_sig.emit(1, "Не удалось открыть 1 устройство")
                return
            cfg = self._config
            # Configure sample rate and baseband filter (fixed 12 MSPS / 8 MHz)
            self._lib.hq_config_set_rates(12000000, 8000000)
            # Configure gains
            self._lib.hq_config_set_gains(cfg.lna_db, cfg.vga_db, cfg.amp_on)
            # Frequency bounds and bin size (clamped to sensible values)
            start_hz = max(cfg.freq_start_hz, 1_000_000)
            stop_hz = min(cfg.freq_end_hz, 6_000_000_000)
            step_hz = max(cfg.bin_hz, 100_000)
            self._lib.hq_config_set_freq_range(float(start_hz), float(stop_hz), float(step_hz))
            # Dwell time in milliseconds (small value for quick sweeps)
            self._lib.hq_config_set_dwell_time(2)
            # Apply detector params
            p = self._parent._detector_params
            self._lib.hq_set_detector_params(
                float(p['threshold_offset']), int(p['min_width']), int(p['min_sweeps']), float(p['timeout'])
            )
            # Grouping tolerance and EMA for smoothing
            self._lib.hq_set_grouping_tolerance_hz(250_000.0)
            self._lib.hq_set_ema_alpha(0.25)
            # Start acquisition.  Try no‑arg form first; fall back to legacy signature.
            try:
                r = self._lib.hq_start()
            except TypeError:
                r = self._lib.hq_start(self._ffi.NULL, self._ffi.NULL)
            if r != 0:
                self._lib.hq_close_all()
                self.finished_sig.emit(3, "Не удалось запустить устройство")
                return
            print(
                f"✅ SingleWorker: Running {int((stop_hz - start_hz) / step_hz)} points, "
                f"range {start_hz/1e6:.1f}-{stop_hz/1e6:.1f} MHz"
            )
            # Determine how many points to request from hq_get_master_spectrum.
            # Ideally we request the full sweep in one call, but limit to
            # 50,000 points to avoid excessive memory usage.  The number of
            # points is (stop_hz - start_hz) / step_hz + 1.
            n_bins = int((stop_hz - start_hz) / step_hz) + 1
            max_points = min(n_bins, 50_000)
            freqs_buf = self._ffi.new("double[]", max_points)
            powers_buf = self._ffi.new("float[]", max_points)
            last_log_time = time.time()
            while self._running and not self._stop_ev.is_set():
                # Read spectrum; blocks until at least one segment is available
                n = self._lib.hq_get_master_spectrum(freqs_buf, powers_buf, max_points)
                if n > 0:
                    freqs = np.frombuffer(self._ffi.buffer(freqs_buf, n * 8), dtype=np.float64, count=n).copy()
                    power = np.frombuffer(self._ffi.buffer(powers_buf, n * 4), dtype=np.float32, count=n).copy()
                    # Discard NaNs or zeroed outputs
                    if not (np.all(np.isnan(power)) or np.all(power == 0)):
                        # Emit the raw segment as a full sweep update for quick UI refresh
                        self._parent.fullSweepReady.emit(freqs, power)
                        self._spectrum_updates += 1
                        # Assemble into a full sweep
                        result = self._assembler.add_segment(freqs, power, int(freqs[0]))
                        if result is not None:
                            full_freqs, full_power = result
                            self._parent.fullSweepReady.emit(full_freqs, full_power)
                        # Periodic logging
                        if time.time() - last_log_time > 5.0:
                            print(
                                f"SingleWorker: {self._spectrum_updates} updates, "
                                f"range {freqs[0]/1e6:.1f}-{freqs[-1]/1e6:.1f} MHz"
                            )
                            last_log_time = time.time()
                # Small delay to limit CPU usage
                # Use small sleep to reduce CPU without impacting latency
                time.sleep(0.02)
        except Exception as e:
            code, msg = 99, str(e)
        finally:
            # Cleanup
            try:
                # Stop acquisition; support both no‑arg and two‑arg forms
                try:
                    self._lib.hq_stop()
                except TypeError:
                    self._lib.hq_stop(self._ffi.NULL, self._ffi.NULL)
                self._lib.hq_close_all()
            except Exception:
                pass
            print(f"SingleWorker stopped after {self._spectrum_updates} spectrum updates")
            self.finished_sig.emit(code, msg)


class _MultiWorker(_BaseWorker):
    """Worker for multi-SDR mode.

    This worker opens multiple devices (typically 3), configures them and
    continuously reads the master spectrum.  Partial sweeps are assembled
    using `_SweepAssembler`.  Recent peaks are polled via
    `hq_get_recent_peaks` and only when peaks are present is the watchlist
    snapshot fetched.  Wideband targets from the watchlist are forwarded as
    `SweepLine` objects via `sweepLine.emit`.  This throttling prevents
    overload of the watchlist and peak queues.
    """

    def __init__(self, parent: HackRFLibSource, ffi: FFI, lib: Any, config: SweepConfig,
                 num_devices: int, assembler: _SweepAssembler) -> None:
        super().__init__()
        self._parent = parent
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._num_devices = num_devices
        self._assembler = assembler
        # Logging counters
        self._spectrum_updates = 0
        self._watchlist_updates = 0

    def run(self) -> None:
        code = 0
        msg = ""
        try:
            # Open the requested number of devices
            if self._lib.hq_open_all(int(self._num_devices)) != 0:
                self.finished_sig.emit(1, f"Не удалось открыть {self._num_devices} устройств")
                return
            cfg = self._config
            # Configure sample rate and baseband filter
            self._lib.hq_config_set_rates(12_000_000, 8_000_000)
            # Configure gains
            self._lib.hq_config_set_gains(cfg.lna_db, cfg.vga_db, cfg.amp_on)
            # Frequency bounds
            start_hz = max(cfg.freq_start_hz, 1_000_000)
            stop_hz = min(cfg.freq_end_hz, 6_000_000_000)
            step_hz = max(cfg.bin_hz, 100_000)
            self._lib.hq_config_set_freq_range(float(start_hz), float(stop_hz), float(step_hz))
            # Dwell time (ms)
            self._lib.hq_config_set_dwell_time(2)
            # Peak grouping tolerance: widen for video signals (5 MHz)
            self._lib.hq_set_grouping_tolerance_hz(5_000_000.0)
            # EMA smoothing factor
            self._lib.hq_set_ema_alpha(0.25)
            # Detector parameters
            p = self._parent._detector_params
            self._lib.hq_set_detector_params(
                float(p['threshold_offset']), int(p['min_width']), int(p['min_sweeps']), float(p['timeout'])
            )
            # Start acquisition.  Try no‑arg form first; fall back to legacy signature.
            try:
                r = self._lib.hq_start()
            except TypeError:
                r = self._lib.hq_start(self._ffi.NULL, self._ffi.NULL)
            if r != 0:
                self._lib.hq_close_all()
                self.finished_sig.emit(3, "Не удалось запустить multi‑SDR")
                return
            print(
                f"✅ MultiWorker: {self._num_devices} devices running. "
                f"Master sweep {start_hz/1e6:.1f}-{stop_hz/1e6:.1f} MHz"
            )
            # Buffers for spectrum and peaks.  Determine the number of points
            # for the master sweep based on the configured range and bin
            # size.  Cap at 50,000 points to limit memory usage.
            n_bins = int((stop_hz - start_hz) / step_hz) + 1
            max_points = min(n_bins, 50_000)
            freqs_buf = self._ffi.new("double[]", max_points)
            powers_buf = self._ffi.new("float[]", max_points)
            peaks_buf = self._ffi.new("Peak[100]")
            watch_buf = self._ffi.new("WatchItem[100]")
            last_log_time = time.time()
            last_watch_time = 0.0
            while self._running and not self._stop_ev.is_set():
                # 1. Master spectrum
                n = self._lib.hq_get_master_spectrum(freqs_buf, powers_buf, max_points)
                if n > 0:
                    freqs = np.frombuffer(self._ffi.buffer(freqs_buf, n * 8), dtype=np.float64, count=n).copy()
                    power = np.frombuffer(self._ffi.buffer(powers_buf, n * 4), dtype=np.float32, count=n).copy()
                    if not np.all(np.isnan(power)):
                        self._parent.fullSweepReady.emit(freqs, power)
                        self._spectrum_updates += 1
                        # Assemble full sweep
                        result = self._assembler.add_segment(freqs, power, int(freqs[0]))
                        if result is not None:
                            full_freqs, full_power = result
                            self._parent.fullSweepReady.emit(full_freqs, full_power)
                # 2. Check peaks every ~100 ms to decide if watchlist should be queried
                if time.time() - last_watch_time > 0.1:
                    last_watch_time = time.time()
                    # Poll only a limited number of recent peaks to avoid
                    # overfilling the internal queue.  Using a small max
                    # improves responsiveness when many peaks are present.
                    n_peaks = self._lib.hq_get_recent_peaks(peaks_buf, 10)
                    if n_peaks > 0:
                        # Only query watchlist when peaks are present
                        # Retrieve only a limited number of watchlist items
                        # to prevent queue overrun; excess items will be
                        # processed in subsequent iterations.
                        n_watch = self._lib.hq_get_watchlist_snapshot(watch_buf, 50)
                        if n_watch > 0:
                            self._watchlist_updates += 1
                            for i in range(n_watch):
                                w = watch_buf[i]
                                # Wideband if bandwidth > 1 MHz and signal strength above noise
                                if w.bw_hz > 1_000_000 and w.rssi_ema > -100:
                                    sw = SweepLine(
                                        ts=None,
                                        f_low_hz=int(w.f_center_hz - w.bw_hz / 2),
                                        f_high_hz=int(w.f_center_hz + w.bw_hz / 2),
                                        bin_hz=int(w.bw_hz / 10) if w.bw_hz > 0 else 200_000,
                                        power_dbm=np.array([w.rssi_ema], dtype=np.float32),
                                    )
                                    # Emit wideband signal for slave targeting and autopicks
                                    self._parent.sweepLine.emit(sw)
                # 3. Periodic status logging
                if time.time() - last_log_time > 5.0:
                    # Query status for active slaves and watch items
                    status = self._ffi.new("HqStatus*")
                    if self._lib.hq_get_status(status) == 0:
                        s = status[0]
                        active = s.slave_running[0] + s.slave_running[1]
                        print(
                            f"MultiWorker: {self._spectrum_updates} spectrum updates, "
                            f"{self._watchlist_updates} watchlist updates, "
                            f"Slaves active: {active}, Targets: {s.watch_items}"
                        )
                    last_log_time = time.time()
                # Avoid busy loop
                time.sleep(0.02)
        except Exception as e:
            code, msg = 99, str(e)
        finally:
            try:
                # Stop acquisition; support both no‑arg and two‑arg forms
                try:
                    self._lib.hq_stop()
                except TypeError:
                    self._lib.hq_stop(self._ffi.NULL, self._ffi.NULL)
                self._lib.hq_close_all()
            except Exception:
                pass
            print(
                f"MultiWorker stopped. Spectrum updates: {self._spectrum_updates}, "
                f"watchlist updates: {self._watchlist_updates}"
            )
            self.finished_sig.emit(code, msg)


class _SweepAssembler:
    """Collects partial spectrum segments and reconstructs full sweeps.

    The HackRF API returns small spectrum segments as the master sweeps the
    configured frequency range.  To present a continuous sweep to the UI and
    to drive the autopicks logic, segments must be aligned on a uniform
    frequency grid and combined.  This class handles the accumulation and
    interpolation of these segments.  When coverage exceeds 80 %, the
    completed sweep is returned and internal state is reset.
    """

    def __init__(self) -> None:
        # Frequency grid parameters
        self.f0_hz = 0
        self.f1_hz = 0
        self.bin_hz = 0
        self.grid: Optional[np.ndarray] = None
        self.n_bins = 0
        # Accumulators
        self.sum: Optional[np.ndarray] = None
        self.cnt: Optional[np.ndarray] = None
        self.seen: Optional[np.ndarray] = None
        self.prev_low: Optional[int] = None

    def configure(self, cfg: SweepConfig) -> None:
        """Initialise the grid from the sweep configuration."""
        self.f0_hz = cfg.freq_start_hz
        self.f1_hz = cfg.freq_end_hz
        self.bin_hz = cfg.bin_hz
        self.grid = np.arange(
            self.f0_hz + self.bin_hz * 0.5,
            self.f1_hz + self.bin_hz * 0.5,
            self.bin_hz,
            dtype=np.float64,
        )
        self.n_bins = len(self.grid)
        self.reset()

    def reset(self) -> None:
        """Reset accumulators for a new sweep."""
        if self.n_bins == 0:
            return
        self.sum = np.zeros(self.n_bins, dtype=np.float64)
        self.cnt = np.zeros(self.n_bins, dtype=np.int32)
        self.seen = np.zeros(self.n_bins, dtype=bool)
        self.prev_low = None

    def add_segment(self, f_hz: np.ndarray, p_dbm: np.ndarray, hz_low: int
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Add a partial segment and return a full sweep when ready.

        Segments are aligned on the internal grid via nearest-neighbour
        mapping.  When coverage reaches at least 80 % of the grid or when
        a wraparound (frequency decreasing) is detected, the accumulated
        sweep is finalised, interpolated and returned.  After returning
        a sweep the assembler resets its accumulators.
        """
        if self.grid is None or self.n_bins == 0:
            return None
        # Detect wraparound (the sweep restarted from the beginning)
        if self.prev_low is not None and hz_low < self.prev_low - 10e6:
            result = self._finalize()
            self.reset()
            self.prev_low = hz_low
            self._add_to_grid(f_hz, p_dbm)
            return result
        # Add segment to grid
        self.prev_low = hz_low
        self._add_to_grid(f_hz, p_dbm)
        coverage = float(self.seen.sum()) / float(self.n_bins) if self.n_bins else 0.0
        # Finalise more eagerly: once we have covered at least 80 % of the grid,
        # produce a sweep.  The remaining gaps will be interpolated.  This
        # improves responsiveness for large sweeps (e.g. 50–6000 MHz) where
        # complete coverage may take many segments.
        if coverage >= 0.80:
            result = self._finalize()
            self.reset()
            return result
        return None

    def _add_to_grid(self, f_hz: np.ndarray, p_dbm: np.ndarray) -> None:
        """Map a segment onto the frequency grid and accumulate power."""
        idx = np.rint((f_hz - self.grid[0]) / self.bin_hz).astype(np.int32)
        mask = (idx >= 0) & (idx < self.n_bins)
        if not np.any(mask):
            return
        idx = idx[mask]
        p = p_dbm[mask].astype(np.float64)
        np.add.at(self.sum, idx, p)
        np.add.at(self.cnt, idx, 1)
        self.seen[idx] = True

    def _finalize(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Finalise the current sweep and return interpolated data."""
        if self.n_bins == 0:
            return None
        coverage = float(self.seen.sum()) / float(self.n_bins)
        if coverage < 0.5:
            return None
        # Average power for bins with data
        p = np.full(self.n_bins, np.nan, dtype=np.float32)
        valid = self.cnt > 0
        p[valid] = (self.sum[valid] / self.cnt[valid]).astype(np.float32)
        # Interpolate missing bins
        if np.isnan(p).any():
            vmask = ~np.isnan(p)
            if vmask.any():
                p = np.interp(np.arange(self.n_bins), np.flatnonzero(vmask), p[vmask]).astype(np.float32)
            p[np.isnan(p)] = -120.0
        return self.grid.copy(), p
