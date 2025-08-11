#!/usr/bin/env python3
import ctypes as C
import argparse, sys, time, threading
from collections import defaultdict

# ---------- настройки по умолчанию ----------
DEFAULT_FREQS_MHZ = [2400, 2450, 2480, 3000, 3500, 5000, 5500, 5800, 6000]
SAMPLES_PER_FREQ = 200
NEIGHBOR_BINS = 3
BIN_HZ = 2500.0
START_STOP_MHZ = (2400, 6000)

# ---------- ctypes сигнатуры ----------
lib = C.CDLL("./libhackrf_qsa.so")

SEGCB = C.CFUNCTYPE(
    None,
    C.POINTER(C.c_double),
    C.POINTER(C.c_float),
    C.c_int,
    C.c_double,
    C.c_uint64,
    C.c_uint64,
    C.c_void_p
)

lib.hq_open.argtypes = [C.c_char_p]
lib.hq_open.restype  = C.c_int
lib.hq_configure.argtypes = [C.c_double, C.c_double, C.c_double, C.c_int, C.c_int, C.c_int]
lib.hq_configure.restype  = C.c_int
lib.hq_start.argtypes = [SEGCB, C.c_void_p]
lib.hq_start.restype  = C.c_int
lib.hq_stop.argtypes = []
lib.hq_stop.restype  = C.c_int
lib.hq_close.argtypes = []
lib.hq_close.restype  = None
lib.hq_last_error.argtypes = []
lib.hq_last_error.restype  = C.c_char_p

# ---------- накопитель ----------
class Accumulator:
    def __init__(self, target_freqs_mhz, samples_per_freq, neighbor_bins):
        self.targets = sorted(target_freqs_mhz)
        self.samples_per_freq = samples_per_freq
        self.neighbor_bins = neighbor_bins
        self.lock = threading.Lock()
        self.data = defaultdict(list)
        self.done = set()

    def want_more(self, f_mhz):
        with self.lock:
            return len(self.data[f_mhz]) < self.samples_per_freq

    def push(self, f_mhz, val):
        with self.lock:
            if len(self.data[f_mhz]) < self.samples_per_freq:
                self.data[f_mhz].append(val)
                if len(self.data[f_mhz]) >= self.samples_per_freq:
                    self.done.add(f_mhz)

    def summary(self):
        import statistics as S
        out = {}
        with self.lock:
            for f in self.targets:
                vals = self.data[f]
                if vals:
                    out[f] = dict(
                        n=len(vals),
                        median=float(S.median(vals)),
                        mean=float(S.mean(vals))
                    )
        return out

# ---------- колбэк ----------
def make_callback(acc: Accumulator, p_known_dbm: float):
    def cb(freqs_ptr, data_ptr, count, bin_w, hz_low, hz_high, user):
        try:
            freqs0 = freqs_ptr[0]
            f_low = hz_low
            f_high = hz_high

            for f_mhz in acc.targets:
                if not acc.want_more(f_mhz):
                    continue
                f0 = f_mhz * 1e6
                if f0 < f_low or f0 > f_high:
                    continue
                idx = int(round((f0 - freqs0) / bin_w))
                if idx < 0 or idx >= count:
                    continue

                k = acc.neighbor_bins
                lo = max(0, idx - k)
                hi = min(count - 1, idx + k)
                s = 0.0
                n = 0
                for i in range(lo, hi + 1):
                    s += data_ptr[i]
                    n += 1
                if n <= 0:
                    continue

                p_meas = s / n
                acc.push(f_mhz, p_meas)
        except Exception:
            pass
    return SEGCB(cb)

# ---------- запись CSV ----------
def write_csv(path, gains, offsets):
    lna, vga, amp = gains
    with open(path, "w", encoding="utf-8") as f:
        f.write("# freq_mhz,lna_db,vga_db,amp,offset_db\n")
        for f_mhz in sorted(offsets.keys()):
            off = offsets[f_mhz]
            f.write(f"{int(round(f_mhz))},{lna},{vga},{amp},{off:+.2f}\n")

# ---------- основной сценарий ----------
def main():
    ap = argparse.ArgumentParser(description="Auto-calibrate HackRF dBm offsets into hackrf_cal.csv")
    ap.add_argument("--serial", default="", help="HackRF serial suffix (optional)")
    ap.add_argument("--lna", type=int, default=32, help="LNA gain (0..40, step 8)")
    ap.add_argument("--vga", type=int, default=40, help="VGA gain (0..62, step 2)")
    ap.add_argument("--amp", type=int, default=0, choices=[0, 1], help="AMP 0/1")
    ap.add_argument("--freqs", default=",".join(map(str, DEFAULT_FREQS_MHZ)),
                    help="Comma-separated target freqs in MHz")
    ap.add_argument("--level", type=float, default=-40.0, help="Known RF level at HackRF input, dBm")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_FREQ, help="Samples per frequency")
    ap.add_argument("--bin", type=float, default=BIN_HZ, help="Requested FFT bin width, Hz")
    ap.add_argument("--out", default="hackrf_cal.csv", help="Output CSV path")
    ap.add_argument("--start", type=float, default=START_STOP_MHZ[0], help="Sweep start MHz")
    ap.add_argument("--stop", type=float, default=START_STOP_MHZ[1], help="Sweep stop MHz")
    args = ap.parse_args()

    targets = [float(x) for x in args.freqs.split(",") if x.strip()]
    acc = Accumulator(targets, args.samples, NEIGHBOR_BINS)

    print("=== Автокалибровка HackRF ===")
    print(f"Профиль: LNA={args.lna} dB  VGA={args.vga} dB  AMP={args.amp}")
    print(f"Уровень генератора: {args.level:.2f} dBm")
    print("Подключи генератор на первую частоту и нажми Enter")
    input()

    if lib.hq_open(args.serial.encode() if args.serial else None) != 0:
        print("hq_open failed:", lib.hq_last_error().decode(), file=sys.stderr)
        return 2

    if lib.hq_configure(args.start, args.stop, args.bin, args.lna, args.vga, args.amp) != 0:
        print("hq_configure failed:", lib.hq_last_error().decode(), file=sys.stderr)
        lib.hq_close()
        return 3

    cb = make_callback(acc, args.level)
    if lib.hq_start(cb, None) != 0:
        print("hq_start failed:", lib.hq_last_error().decode(), file=sys.stderr)
        lib.hq_close()
        return 4

    try:
        for f_mhz in targets:
            if acc.want_more(f_mhz):
                print(f"\n>>> Установи генератор на {int(f_mhz)} МГц и нажми Enter")
                input()
                while acc.want_more(f_mhz):
                    time.sleep(0.2)
                    print(f"  собираю {int(f_mhz)} МГц: {len(acc.data[f_mhz])}/{args.samples}", end="\r")
        print("\nСбор завершён.")
    finally:
        lib.hq_stop()
        lib.hq_close()

    import statistics as S
    offsets = {}
    for f_mhz, stats in acc.summary().items():
        k = args.level - stats["median"]
        offsets[f_mhz] = k
        print(f"{int(f_mhz)} МГц: offset {k:+.2f} dB")

    write_csv(args.out, (args.lna, args.vga, args.amp), offsets)
    print(f"Готово. Сохранено в {args.out}")

if __name__ == "__main__":
    sys.exit(main())
