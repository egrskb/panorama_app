# hq_cffi.py
from cffi import FFI
import os, sys
ffi = FFI()

ffi.cdef(r"""
typedef void (*hq_segment_cb)(const double*, const float*, int, double, uint64_t, uint64_t, void*);
int  hq_open(const char* serial_suffix);
int  hq_configure(double f_start_mhz, double f_stop_mhz, double requested_bin_hz,
                  int lna_db, int vga_db, int amp_enable);
int hq_device_count(void);
int hq_get_device_serial(int idx, char* buf, int buf_len);
int  hq_start(hq_segment_cb cb, void* user);
int  hq_stop(void);
void hq_close(void);
const char* hq_last_error(void);
""")


def list_hackrf_serials():
    ser = []
    try:
        n = lib.hq_device_count()
        for i in range(int(n)):
            b = ffi.new("char[128]")
            if lib.hq_get_device_serial(i, b, 127) == 0:
                s = ffi.string(b).decode()
                if s: ser.append(s)
    except Exception:
        pass
    return ser



def _lib_path():
    # подхватываем рядом с exe/скриптом либо из LD_LIBRARY_PATH
    names = ["libhackrf_qsa.so", "libhackrf_qsa.dylib", "hackrf_qsa.dll"]
    here = os.path.abspath(os.path.dirname(__file__))
    for n in names:
        p = os.path.join(here, n)
        if os.path.exists(p): return p
    # fallback — пусть dlopen сам ищет
    return names[0]

lib = ffi.dlopen(_lib_path())