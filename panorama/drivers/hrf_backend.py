#!/usr/bin/env python3
"""
HackRF Master backend (CFFI) для панорамы RSSI.

- Экспортирует класс HackRFMaster (как и ожидает менеджер устройств).
- Экспортирует enumerate_devices() на уровне модуля.
- Внутри использует CFFI-обёртку к libhackrf_qsa.so и индексную склейку свипов
  через SweepAssembler (центры бинов, глобальная сетка).
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
# ОСНОВНОЙ БЭКЕНД (QSA)
# ------------------------------------------------------------

class HackRFQSABackend(SourceBackend):
    """Работа с HackRF через libhackrf_qsa.so, отдаёт полные свипы в GUI."""

    def __init__(self, serial_suffix: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._ffi, self._lib = _load_lib()
        self._serial = (serial_suffix or "").strip()
        self._worker: Optional[_Worker] = None
        self._assembler = SweepAssembler()

    # ---------- управление ----------
    def start(self, config: SweepConfig):
        if self.is_running():
            return

        # Настраиваем сборщик (индексная укладка по глобальной сетке центров бинов)
        self._assembler.configure(config.freq_start_hz, config.freq_end_hz, config.bin_hz)

        self._worker = _Worker(self, self._ffi, self._lib, config, self._serial, self._assembler)
        self._worker.finished_sig.connect(self._on_finished)
        self._worker.start()
        self.started.emit()

    def stop(self):
        if not self.is_running():
            return
        try:
            self._lib.hq_stop()
        except Exception:
            pass
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
        try:
            ffi, lib = _load_lib()
            n = int(lib.hq_device_count())
            out: List[str] = []
            for i in range(n):
                buf = ffi.new("char[128]")
                rc = int(lib.hq_get_device_serial(i, buf, 127))
                if rc == 0:
                    s = ffi.string(buf).decode(errors="ignore").strip()
                    # фильтруем мусор
                    if s and s != "0000000000000000":
                        out.append(s)
            return out
        except Exception as e:
            print(f"[HackRF] enumerate_devices failed: {e}")
            return []


# ------------------------------------------------------------
# СОВМЕСТИМОСТЬ С СТАРЫМ ИМЕНЕМ
# ------------------------------------------------------------

class HackRFMaster(HackRFQSABackend):
    """Старое ожидаемое именование бэкенда — оставлено для совместимости."""
    pass


# ------------------------------------------------------------
# ВНУТРЕННИЙ РАБОЧИЙ ПОТОК
# ------------------------------------------------------------

class _Worker(QtCore.QThread):
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

                row, coverage = self._assembler.feed({
                    "freqs_hz": freqs,
                    "data_dbm": power,
                    "hz_low": float(f_low_hz),
                })
                if row is not None:
                    nb = self._assembler.nbins
                    f0 = self._assembler.f0
                    b  = self._assembler.bin_hz
                    full_freqs = (f0 + (np.arange(nb, dtype=np.float64) + 0.5) * b).astype(np.float64)
                    full_power = row.astype(np.float32)
                    self._backend.fullSweepReady.emit(full_freqs, full_power)
                    # чуть притормозим, чтобы UI не задыхался
                    time.sleep(0.003)
            except Exception as e:
                print(f"[HackRF] callback error: {e}")

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
                    # 2) пробуем суффиксы (часто прошивка ждёт suffix)
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
            r = int(self._lib.hq_configure(
                float(cfg.freq_start_hz) / 1e6,
                float(cfg.freq_end_hz) / 1e6,
                float(cfg.bin_hz),
                int(cfg.lna_db),
                int(cfg.vga_db),
                int(1 if cfg.amp_on else 0),
            ))
            if r != 0:
                last = self._lib.hq_last_error()
                last_msg = self._ffi.string(last).decode(errors="ignore") if last != self._ffi.NULL else ""
                msg = f"hq_configure failed ({r}); {last_msg}"
                self._lib.hq_close()
                self.finished_sig.emit(2, msg)
                return

            # --- СТАРТ ---
            r = int(self._lib.hq_start(self._cb, self._ffi.NULL))
            if r != 0:
                last = self._lib.hq_last_error()
                last_msg = self._ffi.string(last).decode(errors="ignore") if last != self._ffi.NULL else ""
                msg = f"hq_start failed ({r}); {last_msg}"
                self._lib.hq_close()
                self.finished_sig.emit(3, msg)
                return

            # --- ЦИКЛ ---
            while not self._stop_event.is_set():
                time.sleep(0.05)

            # --- СТОП ---
            self._lib.hq_stop()
            self._lib.hq_close()
            self.finished_sig.emit(0, "")

        except Exception as e:
            try:
                last = self._lib.hq_last_error()
                last_msg = self._ffi.string(last).decode(errors="ignore") if last != self._ffi.NULL else ""
            except Exception:
                last_msg = ""
            msg = last_msg or str(e)
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
# МОДУЛЬНЫЙ API, КОТОРЫЙ ЖДЁТ МЕНЕДЖЕР УСТРОЙСТВ
# ------------------------------------------------------------

def enumerate_devices() -> List[str]:
    """Публичная функция, которую импортирует Device Manager."""
    return HackRFQSABackend.enumerate_devices()
