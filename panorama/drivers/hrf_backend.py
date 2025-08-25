#!/usr/bin/env python3
"""
HackRF Master backend (CFFI) для панорамы RSSI - OPTIMIZED VERSION.
Исправлена инициализация SweepAssembler.
"""

from __future__ import annotations
from typing import Optional, List
import time
import threading
import numpy as np
from pathlib import Path
from cffi import FFI
from PyQt5 import QtCore

from panorama.drivers.base import SourceBackend, SweepConfig
from panorama.features.spectrum.service import SweepAssembler


# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНОЕ: загрузка .so и C-интерфейс
# ------------------------------------------------------------

def _load_lib():
    ffi = FFI()
    ffi.cdef(r"""
        typedef void (*hq_segment_cb)(
            const double* freqs_hz,
            const float*  data_dbm,
            int count,
            double fft_bin_width_hz,
            uint64_t hz_low, uint64_t hz_high,
            void* user
        );

        int  hq_open(const char* serial_suffix_or_null);
        void hq_close(void);

        int  hq_configure(double f_start_mhz, double f_stop_mhz, double bin_hz,
                          int lna_db, int vga_db, int amp_on);

        int  hq_start(hq_segment_cb cb, void* user);
        int  hq_stop(void);

        const char* hq_last_error(void);

        int  hq_device_count(void);
        int  hq_get_device_serial(int idx, char* out, int cap);
    """)

    lib_name = "libhackrf_qsa.so"
    search_dirs = [
        Path(__file__).resolve().parent / "hackrf_master" / "build",
        Path(__file__).resolve().parent / "hackrf_master",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]
    lib_path = None
    for d in search_dirs:
        p = d / lib_name
        if p.exists():
            lib_path = str(p)
            break
    if lib_path is None:
        lib_path = lib_name

    lib = ffi.dlopen(lib_path)
    return ffi, lib


# ------------------------------------------------------------
# ОСНОВНОЙ БЭКЕНД (QSA) - OPTIMIZED VERSION
# ------------------------------------------------------------

class HackRFQSABackend(SourceBackend):
    """Работа с HackRF через libhackrf_qsa.so, отдаёт полные свипы в GUI."""

    def __init__(self, serial_suffix: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._ffi, self._lib = _load_lib()
        self._serial = (serial_suffix or "").strip()
        self._worker: Optional[_Worker] = None
        # НЕ создаем SweepAssembler здесь - создадим в start() с правильными параметрами
        self._assembler: Optional[SweepAssembler] = None

    # ---------- управление ----------
    def start(self, config: SweepConfig):
        if self.is_running():
            return

        # Создаем и настраиваем сборщик с правильными параметрами
        self._assembler = SweepAssembler(coverage_threshold=0.45, wrap_guard_hz=100e6)
        self._assembler.configure(config.freq_start_hz, config.freq_end_hz, config.bin_hz)

        self._worker = _Worker(self, self._ffi, self._lib, config, self._serial, self._assembler)
        self._worker.finished_sig.connect(self._on_finished)
        self._worker.start()

        self.started.emit()

    def stop(self):
        if not self.is_running():
            return

        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @QtCore.pyqtSlot(int, str)
    def _on_finished(self, code: int, message: str):
        if code != 0:
            self.error.emit(message or "Unknown error")
        self.finished.emit(code)

    # ---------- перечисление устройств ----------
    @staticmethod
    def enumerate_devices() -> List[str]:
        """Возвращает список серийников HackRF."""
        try:
            ffi = FFI()
            ffi.cdef(r"""
                int  hq_device_count(void);
                int  hq_get_device_serial(int idx, char* out, int cap);
            """)
            try:
                lib = ffi.dlopen("libhackrf_qsa.so")
                n = lib.hq_device_count()
                devices: List[str] = []
                for i in range(int(n)):
                    buf = ffi.new("char[128]")
                    ok = lib.hq_get_device_serial(i, buf, 127)
                    if int(ok) == 0:
                        serial = ffi.string(buf).decode(errors="ignore")
                        if serial and serial != "0000000000000000":
                            devices.append(serial)
                return devices
            except Exception as e:
                print(f"[HackRFQSA] enumerate_devices failed: {e}")
                return []
        except Exception:
            return []


# ------------------------------------------------------------
# ВНУТРЕННИЙ РАБОЧИЙ ПОТОК - OPTIMIZED VERSION
# ------------------------------------------------------------

class _Worker(QtCore.QThread):
    """Рабочий поток: открывает/конфигурирует устройство и обрабатывает callback-и."""

    finished_sig = QtCore.pyqtSignal(int, str)

    def __init__(self, backend: HackRFQSABackend, ffi: FFI, lib, config: SweepConfig,
                 serial: str, assembler: SweepAssembler):
        super().__init__()
        self._backend = backend
        self._ffi = ffi
        self._lib = lib
        self._config = config
        self._serial = serial
        self._assembler = assembler
        self._stop_event = threading.Event()
        
        # OPTIMIZED: более мягкие настройки throttling
        self._last_emit_time = 0.0
        self._min_emit_interval = 0.1  # 10 FPS максимум
        self._segment_counter = 0
        self._max_segments_before_emit = 50  # эмитим после 50 сегментов
        self._last_reset_time = time.time()
        self._reset_interval = 5.0  # сбрасываем цикл каждые 5 секунд
        self._block_counter = 0

        @self._ffi.callback("void(const double*, const float*, int, double, uint64_t, uint64_t, void*)")
        def _cb(freqs_ptr, pwr_ptr, n_bins, fft_bin_width_hz, f_low_hz, f_high_hz, user):
            if self._stop_event.is_set():
                return
            try:
                n = int(n_bins)
                if n <= 0:
                    return
                freqs = np.frombuffer(self._ffi.buffer(freqs_ptr, n * 8), dtype=np.float64, count=n).copy()
                power = np.frombuffer(self._ffi.buffer(pwr_ptr, n * 4), dtype=np.float32, count=n).copy()
                if np.all(np.isnan(power)) or np.all(power == 0):
                    return

                # Кладём сегмент в ассемблер
                row, coverage = self._assembler.feed({
                    "freqs_hz": freqs,
                    "data_dbm": power,
                    "hz_low": float(f_low_hz),
                })

                # Увеличиваем счетчики
                self._segment_counter += 1
                self._block_counter += 1
                current_time = time.time()

                # Проверяем, нужно ли сбросить цикл по таймауту
                if current_time - self._last_reset_time > self._reset_interval and coverage < 0.3:
                    self._assembler.reset_pass()
                    self._last_reset_time = current_time
                    self._segment_counter = 0
                    print(f"[HackRF] Auto-reset cycle after {self._reset_interval}s timeout")

                # Логирование прогресса каждые 100 блоков
                if self._block_counter % 100 == 0:
                    print(f"[hackrf_master] Processed {self._block_counter} blocks")

                # Эмитим при готовности полной строки или достаточном накоплении
                if row is not None and current_time - self._last_emit_time >= self._min_emit_interval:
                    nb = self._assembler.nbins
                    f0 = self._assembler.f0
                    b  = self._assembler.bin_hz
                    full_freqs = (f0 + (np.arange(nb, dtype=np.float64) + 0.5) * b).astype(np.float64)
                    full_power = row.astype(np.float32, copy=False)
                    
                    self._backend.fullSweepReady.emit(full_freqs, full_power)
                    
                    self._last_emit_time = current_time
                    self._segment_counter = 0
                    
                    print(f"[HackRF] Full sweep ready: N={len(full_freqs)}, "
                          f"{full_freqs[0]/1e6:.3f}-{full_freqs[-1]/1e6:.3f} MHz, "
                          f"coverage={coverage:.3f}")

            except Exception as e:
                print(f"[HackRF] callback error: {e}")
                import traceback
                print(traceback.format_exc())

        self._cb = _cb

    def run(self):
        code = 0
        msg = ""
        try:
            # --- ОТКРЫТИЕ УСТРОЙСТВА С ФОЛЛБЭКАМИ ---
            serial_c = None
            tried_serials = []

            def _try_open(s: Optional[str]) -> int:
                c = self._ffi.NULL if not s else self._ffi.new("char[]", s.encode("utf-8"))
                return int(self._lib.hq_open(c))

            # 1) пробуем как есть (полный серийник из конфига)
            s = self._serial or ""
            if s:
                tried_serials.append(s)
                r = _try_open(s)
                if r != 0:
                    # 2) пробуем суффиксы
                    for suf_len in (16, 12, 8):
                        if len(s) >= suf_len:
                            suf = s[-suf_len:]
                            if suf not in tried_serials:
                                tried_serials.append(suf)
                                r = _try_open(suf)
                                if r == 0:
                                    break
                if r != 0:
                    # 3) без серийника — первое доступное устройство
                    if "" not in tried_serials:
                        tried_serials.append("")
                        r = _try_open(None)
            else:
                # с пустым серийником: сразу пытаемся открыть первое доступное
                tried_serials.append("")
                r = _try_open(None)

            if r != 0:
                # выдадим подробное сообщение
                last = self._lib.hq_last_error()
                last_msg = ""
                if last != self._ffi.NULL:
                    last_msg = self._ffi.string(last).decode(errors="ignore")
                msg = f"hq_open failed ({r}); tried={tried_serials}; {last_msg}"
                self.finished_sig.emit(1, msg)
                return

            # --- КОНФИГУРАЦИЯ ---
            cfg = self._config
            r = self._lib.hq_configure(
                float(cfg.freq_start_hz / 1e6),
                float(cfg.freq_end_hz / 1e6),
                float(cfg.bin_hz),
                int(cfg.lna_db),
                int(cfg.vga_db),
                int(1 if cfg.amp_on else 0)
            )
            if int(r) != 0:
                msg = f"hq_configure failed ({int(r)})"
                self._lib.hq_close()
                self.finished_sig.emit(2, msg)
                return

            # --- ЗАПУСК SWEEP ---
            r = self._lib.hq_start(self._cb, self._ffi.NULL)
            if int(r) != 0:
                msg = f"hq_start failed ({int(r)})"
                self._lib.hq_close()
                self.finished_sig.emit(3, msg)
                return

            print(f"[HackRF] Started sweep: {cfg.freq_start_hz/1e6:.1f}-{cfg.freq_end_hz/1e6:.1f} MHz, "
                  f"bin={cfg.bin_hz:.0f} Hz")

            # --- РАБОЧИЙ ЦИКЛ ---
            while not self._stop_event.is_set():
                time.sleep(0.1)  # проверяем остановку каждые 100ms

            # --- ОСТАНОВКА ---
            self._lib.hq_stop()
            self._lib.hq_close()
            self.finished_sig.emit(0, "")

        except Exception as e:
            try:
                last = self._lib.hq_last_error()
                if last != self._ffi.NULL:
                    msg = self._ffi.string(last).decode(errors="ignore")
                else:
                    msg = str(e)
            except Exception:
                msg = str(e)
            code = 4
            try:
                self._lib.hq_stop()
                self._lib.hq_close()
            except Exception:
                pass
            self.finished_sig.emit(code, msg)

    def stop(self):
        self._stop_event.set()


# ------------------------------------------------------------
# СОВМЕСТИМОСТЬ
# ------------------------------------------------------------

class HackRFMaster(HackRFQSABackend):
    """Старое ожидаемое именование бэкенда — оставлено для совместимости."""
    pass


# ------------------------------------------------------------
# МОДУЛЬНЫЙ API
# ------------------------------------------------------------

def enumerate_devices() -> List[str]:
    """
    ВАЖНО: публичный API, который импортирует менеджер устройств.
    Возвращает список серийников HackRF.
    """
    return HackRFQSABackend.enumerate_devices()