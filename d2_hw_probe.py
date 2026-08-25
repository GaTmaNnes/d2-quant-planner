#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 HW PROBE — fingerprint matériel mesuré (pas seulement lu).
=============================================================
Couche D2-HW-PROFILER indépendante du matériel (spec D2, sections 1-2, 6).

Mesure, sans charge de modèle :
  - CPU     : modèle, coeurs logiques/physiques, fréquence, cache L1/L2/L3, SIMD (heuristique)
  - RAM     : capacité, canaux, bande passante MESURÉE (copy multithread numpy)
  - GPU     : nom, driver, VRAM, sm, support FP8/INT8/FP16 (déduit de sm)
  - VRAM    : bande passante MESURÉE (cudaMemcpy D2D via cudart)
  - PCIe    : génération/largeur (nvidia-smi) + courbe de transport MESURÉE H2D/D2H
  - DMA     : latence petit transfert + copie concurrente + overlap copy/compute
  - compute : GEMM FP32/FP16 MESURÉ (cublas) + GEMM CPU numpy

Sortie : hardware_fingerprint.json (schéma D2_PROFILE / MACHINE + MICROBENCH).

Exemples :
  python d2_hw_probe.py
  python d2_hw_probe.py --json d2_hw_fingerprint.json --sizes "1k,4k,16k,64k,256k,1m,4m,16m,64m,256m,1g"
  python d2_hw_probe.py --no-cuda   # CPU/RAM/GPU(nvidia-smi) uniquement
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import threading
import time
import ctypes
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NPU_METRICS_URL = "http://127.0.0.1:8765/metrics"


def _find_cuda_dll(basename_prefix):
    """Trouve la DLL CUDA la plus recente disponible (cudart64_12.dll,
    cudart64_13.dll, ...) au lieu d'un numero de version fige en dur, qui se
    perime a chaque changement de toolkit CUDA (ex: build sm_120 en CUDA 13
    a remplace les cudart64_12.dll/cublas64_12.dll a la racine)."""
    candidates = []
    for base in (HERE, os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")):
        candidates += glob.glob(os.path.join(base, f"{basename_prefix}_*.dll"))
    def _version(path):
        name = os.path.basename(path)
        digits = "".join(ch for ch in name.split("_")[-1] if ch.isdigit())
        return int(digits) if digits else -1
    candidates.sort(key=_version, reverse=True)
    return candidates

MEMCPY_KIND = {"H2H": 0, "H2D": 1, "D2H": 2, "D2D": 3}

DEFAULT_SIZES = [1024, 4096, 16384, 65536, 262144,
                 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30]

SIZE_LABELS = ["1KB", "4KB", "16KB", "64KB", "256KB", "1MB",
               "4MB", "16MB", "64MB", "256MB", "1GB"]


def run(cmd, timeout=40):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return -1, str(e)


def ps(cmd, timeout=40):
    rc, out = run(cmd, timeout=timeout)
    return out if rc == 0 else None


def first_int(s, default=None):
    if not s:
        return default
    import re
    m = re.search(r"\d+", s)
    return int(m.group()) if m else default


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
def probe_cpu():
    cpu = {
        "vendor_model": platform.processor() or "unknown",
        "architecture": platform.machine(),
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "frequency_mhz": None,
        "cache_l1_kb": None,
        "cache_l2_kb": None,
        "cache_l3_kb": None,
        "simd": None,
        "source": "spec",
    }
    try:
        import psutil
        cpu["physical_cores"] = psutil.cpu_count(logical=False)
        f = psutil.cpu_freq()
        if f:
            cpu["frequency_mhz"] = int(f.current or f.max)
    except Exception:
        pass

    name = ps("wmic cpu get name /value")
    if name:
        for line in name.splitlines():
            if "=" in line:
                cpu["vendor_model"] = line.split("=", 1)[1].strip()

    cache = ps("wmic cpu get L1CacheSize,L2CacheSize,L3CacheSize /value")
    if cache:
        for line in cache.splitlines():
            k, _, v = line.partition("=")
            k = k.strip()
            if v.strip().isdigit():
                kb = int(v.strip())
                if "L1" in k:
                    cpu["cache_l1_kb"] = kb
                elif "L2" in k:
                    cpu["cache_l2_kb"] = kb
                elif "L3" in k:
                    cpu["cache_l3_kb"] = kb

    cpu["simd"] = _guess_simd(cpu["vendor_model"])
    return cpu


def _guess_simd(model):
    m = (model or "").upper()
    if "AVX-512" in m or "AVX512" in m:
        return "AVX512"
    if "AVX2" in m:
        return "AVX2"
    if "AVX" in m:
        return "AVX"
    # Détection réelle via l'API Windows (IsProcessorFeaturePresent), fiable sur
    # Windows 10/11 (ex. Ryzen AI 9 365 = Zen 5 -> AVX512). Retombe sur l'heuristique
    # de nom hors Windows.
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        if k32.IsProcessorFeaturePresent(41):   # PF_AVX512F_INSTRUCTIONS_AVAILABLE
            return "AVX512"
        if k32.IsProcessorFeaturePresent(40):   # PF_AVX2_INSTRUCTIONS_AVAILABLE
            return "AVX2"
        if k32.IsProcessorFeaturePresent(39):   # PF_AVX_INSTRUCTIONS_AVAILABLE
            return "AVX"
    except Exception:
        pass
    return "SSE" if "X64" in m or "INTEL" in m or "AMD" in m else "unknown"


# ---------------------------------------------------------------------------
# RAM (mesure réelle)
# ---------------------------------------------------------------------------
def probe_ram():
    ram = {"capacity_gb": None, "channels": None, "theoretical_bw_gbs": None,
           "measured_bw_gbs": None, "source": "spec"}
    try:
        import psutil
        ram["capacity_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass

    chips = ps("wmic memorychip get Capacity,ConfiguredClockSpeed /format:list")
    n_chips = 0
    speed = None
    if chips:
        for line in chips.splitlines():
            k, _, v = line.partition("=")
            k = k.strip()
            if k == "Capacity" and v.strip().isdigit():
                n_chips += 1
            elif k == "ConfiguredClockSpeed" and v.strip().isdigit():
                speed = int(v.strip())
    if n_chips:
        ram["channels"] = n_chips
        if speed:
            # ConfiguredClockSpeed est déjà le débit EFFECTIF (MT/s) pour la DDR :
            # ne PAS re-multiplier par 2 (double comptage du facteur DDR).
            # Ex. DDR5-5600 double canal : 2 x 5600 MT/s x 8 o = 89,6 Go/s.
            ram["theoretical_bw_gbs"] = round(n_chips * speed * 8 / 1000, 1)

    try:
        ram["measured_bw_gbs"] = _bench_ram_bw()
        ram["source"] = "measure"
    except Exception as e:
        ram["bw_error"] = str(e)
    return ram


def _bench_ram_bw(size_mb=1024, repeats=8):
    """Bande passante RAM façon STREAM Copy (b[:] = a[:]).

    L'ancienne version recreait `nthreads` threads Python a chaque repetition
    sur un buffer de seulement 256 Mo : le cout de demarrage des threads
    (~qq centaines de us x8, sur Windows) et la taille reduite du transfert
    rendaient la mesure dominee par l'overhead, pas par la memoire — d'ou un
    resultat ~32 GB/s alors que la config (DDR5 dual-channel, 179 GB/s
    theorique) devrait tenir un debit nettement plus eleve. Corrige via :
    - un pool de threads reutilise (cree une seule fois, pas par repetition),
    - un buffer plus gros (1 Go) pour amortir l'overhead de dispatch,
    - une passe de warm-up hors chronometrage (premier touch des pages),
    - le calcul en octets *deplaces* (lecture + ecriture), convention STREAM,
      au lieu de compter seulement les octets source (qui sous-estimait la
      bande passante reelle d'un facteur 2).
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    n = int(size_mb * 1e6 // 8)
    a = np.ones(n, dtype=np.float64)
    b = np.empty_like(a)
    nthreads = min((os.cpu_count() or 4), 8)
    chunk = n // nthreads
    bounds = [(i * chunk, n if i == nthreads - 1 else (i + 1) * chunk) for i in range(nthreads)]

    def _cp(bound):
        s, e = bound
        b[s:e] = a[s:e]

    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        # warm-up : premier touch des pages, hors chronometrage
        list(pool.map(_cp, bounds))

        best = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            list(pool.map(_cp, bounds))
            dt = time.perf_counter() - t0
            bytes_moved = a.nbytes * 2  # lecture (a) + ecriture (b)
            gbs = bytes_moved / dt / 1e9
            best = max(best, gbs)
    return round(best, 1)


# ---------------------------------------------------------------------------
# GPU / VRAM / PCIe / DMA (nvidia-smi + cudart via ctypes)
# ---------------------------------------------------------------------------
def probe_npu():
    """NPU (AMD Ryzen AI / XDNA2 Strix) via le capteur de telemetrie local
    (:8765/metrics, meme source que llama_ui.py). Mesure reelle rapportee
    par ce capteur (bande passante, etat, goulot) : pas une lecture de specs
    vendeur, pas une estimation. Si le capteur ne tourne pas sur ce poste,
    on ne fabrique aucune valeur : "present": False, "source": "unavailable".
    """
    npu = {"present": False, "source": "unavailable"}
    try:
        with urllib.request.urlopen(NPU_METRICS_URL, timeout=1.5) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        npu["error"] = str(e)
        return npu

    n = data.get("npu") or {}
    if not n:
        return npu

    npu.update({
        "present": bool(n.get("available", False)),
        "source": "measure",
        "device": n.get("device"),
        "state": n.get("state"),
        "bottleneck": n.get("bottleneck"),
        "bandwidth_gbs": n.get("bw_gbs"),
        "load_pct": n.get("load_pct"),
        "mem_mb": n.get("mem_mb"),
        "temp_c": n.get("temp_c"),
        "power_w": n.get("power_w"),
        "freq_mhz": n.get("freq_mhz"),
        "tokens_per_sec": n.get("tps"),
        "dispatch_s": n.get("dispatch_s"),
        "sync_us": n.get("sync_us"),
        "mac_eff_pct": n.get("mac_eff_pct"),
        "gops": n.get("gops"),
        "errors": n.get("errors"),
        "note": f"capteur local {NPU_METRICS_URL}",
    })
    return npu


def probe_gpu_smi():
    gpu = {"present": False, "source": "spec"}
    rc, out = run("nvidia-smi --query-gpu=name,memory.total,driver_version,"
                  "compute_cap,clocks.max.sm,clocks.max.mem "
                  "--format=csv,noheader,nounits", timeout=20)
    if rc != 0 or not out:
        return gpu
    # une ligne par GPU : prendre la première (les virgules des lignes suivantes
    # casseraient le découpage quand plusieurs GPU sont présents).
    first_line = out.splitlines()[0]
    parts = [x.strip() for x in first_line.split(",")]
    if len(parts) < 6:
        return gpu
    gpu.update({
        "present": True,
        "name": parts[0],
        "vram_mib": first_int(parts[1]),
        "driver": parts[2],
        "compute_cap": parts[3],
        "max_sm_mhz": first_int(parts[4]),
        "max_mem_mhz": first_int(parts[5]),
    })
    sm = 0
    try:
        sm = int(parts[3].replace(".", ""))
    except Exception:
        sm = 0
    gpu["fp16_hw"] = sm >= 61
    gpu["bf16_hw"] = sm >= 80
    gpu["int8_hw"] = sm >= 61  # DP4A (INT8) disponible dès Pascal sm_61
    gpu["fp8_hw"] = sm >= 89
    gpu["fp4_hw"] = sm >= 100
    gpu["arch"] = "Blackwell" if sm >= 100 else ("Ada" if sm >= 89 else
                  ("Hopper" if sm >= 90 else ("Ampere" if sm >= 80 else
                   ("Turing" if sm >= 75 else ("Volta" if sm >= 70 else "Pascal")))))
    rc, pcie = run("nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max,"
                   "pcie.link.gen.current,pcie.link.width.current "
                   "--format=csv,noheader,nounits", timeout=20)
    if rc == 0 and pcie:
        g = [x.strip() for x in pcie.splitlines()[0].split(",")]
        if len(g) >= 4:
            gpu["pcie_max_gen"] = first_int(g[0])
            gpu["pcie_max_width"] = first_int(g[1])
            gpu["pcie_cur_gen"] = first_int(g[2])
            gpu["pcie_cur_width"] = first_int(g[3])
    return gpu


class CudaRuntime:
    """Wrapper ctypes minimal sur cudart64_12.dll (runtime API CUDA)."""

    def __init__(self, dll_paths):
        self.lib = None
        for p in dll_paths:
            if os.path.exists(p):
                try:
                    self.lib = ctypes.CDLL(p)
                    break
                except OSError:
                    continue
        if self.lib is None:
            raise OSError("cudart introuvable")

        self._errstr = self.lib.cudaGetErrorString
        self._errstr.argtypes = [ctypes.c_int]
        self._errstr.restype = ctypes.c_char_p

        for fn, args in [
            ("cudaGetDeviceCount", [ctypes.POINTER(ctypes.c_int)]),
            ("cudaSetDevice", [ctypes.c_int]),
            ("cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
            ("cudaFree", [ctypes.c_void_p]),
            ("cudaMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
            ("cudaMemcpyAsync", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]),
            ("cudaMemsetAsync", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]),
            ("cudaEventCreate", [ctypes.POINTER(ctypes.c_void_p)]),
            ("cudaEventRecord", [ctypes.c_void_p, ctypes.c_void_p]),
            ("cudaEventSynchronize", [ctypes.c_void_p]),
            ("cudaEventElapsedTime", [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]),
            ("cudaStreamCreate", [ctypes.POINTER(ctypes.c_void_p)]),
            ("cudaStreamSynchronize", [ctypes.c_void_p]),
        ]:
            f = getattr(self.lib, fn)
            f.argtypes = [a for a in args]
            f.restype = ctypes.c_int

        self.n_dev = ctypes.c_int(0)
        self.check(self.lib.cudaGetDeviceCount(ctypes.byref(self.n_dev)))
        if self.n_dev.value < 1:
            raise OSError("aucun device CUDA")

    def check(self, code):
        if code != 0:
            raise OSError(self._errstr(code).decode("utf-8", "replace"))

    def alloc(self, size):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), size))
        return p

    def free(self, p):
        self.check(self.lib.cudaFree(p))

    def memcpy(self, dst, src, size, kind):
        self.check(self.lib.cudaMemcpy(dst, src, size, kind))

    def timed(self, fn):
        e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(e0)))
        self.check(self.lib.cudaEventCreate(ctypes.byref(e1)))
        ms = ctypes.c_float(0)
        self.check(self.lib.cudaEventRecord(e0, None))
        fn()
        self.check(self.lib.cudaEventRecord(e1, None))
        self.check(self.lib.cudaEventSynchronize(e1))
        self.check(self.lib.cudaEventElapsedTime(ctypes.byref(ms), e0, e1))
        self.lib.cudaEventDestroy(e0)
        self.lib.cudaEventDestroy(e1)
        return ms.value


def bench_transfer_curve(cuda, sizes, kind, repeats=3):
    """Courbe de transport : (size_bytes, bw_gbs, ms) pour un sens donné."""
    out = []
    for size in sizes:
        best = None
        try:
            h = (ctypes.c_ubyte * size)()
            d = cuda.alloc(size)
            for _ in range(repeats):
                if kind == MEMCPY_KIND["H2D"]:
                    ms = cuda.timed(lambda: cuda.memcpy(d, h, size, kind))
                else:
                    ms = cuda.timed(lambda: cuda.memcpy(h, d, size, kind))
                bw = size / max(ms, 1e-9) / 1e6
                if best is None or ms < best[1]:
                    best = (bw, ms)
            cuda.free(d)
        except Exception as e:
            out.append({"size": size, "label": _label(size), "error": str(e)})
            continue
        out.append({"size": size, "label": _label(size),
                    "bw_gbs": round(best[0], 2), "ms": round(best[1], 4)})
    return out


def _label(size):
    g = size / (1 << 30)
    if g >= 1:
        return f"{g:.0f}GB"
    m = size / (1 << 20)
    if m >= 1:
        return f"{m:.0f}MB"
    k = size / 1024
    return f"{k:.0f}KB" if k >= 1 else f"{size}B"


def bench_d2d(cuda, sizes):
    out = []
    for size in sizes:
        try:
            a, b = cuda.alloc(size), cuda.alloc(size)
            best = 1e18
            for _ in range(3):
                ms = cuda.timed(lambda: cuda.memcpy(b, a, size, MEMCPY_KIND["D2D"]))
                best = min(best, ms)
            out.append({"size": size, "label": _label(size),
                        "bw_gbs": round(size / best / 1e6, 2), "ms": round(best, 4)})
            cuda.free(a)
            cuda.free(b)
        except Exception as e:
            out.append({"size": size, "label": _label(size), "error": str(e)})
    return out


def bench_memset_write(cuda, size=1 << 30, reps=3):
    """Bande passante d'ECRITURE VRAM : cudaMemsetAsync (moteur DMA, remplit size octets)."""
    d = cuda.alloc(size)
    best = 0.0
    try:
        for _ in range(reps):
            ms = cuda.timed(lambda: cuda.check(cuda.lib.cudaMemsetAsync(d, 0, size, None)))
            bw = size / max(ms, 1e-9) / 1e6
            best = max(best, bw)
    finally:
        cuda.free(d)
    return round(best, 2)


def _optimal_transfer(curve):
    """Plus petite taille de transfert atteignant >= 90 % du pic de la courbe.

    Le PIC est le maximum de la courbe (pas la dernière valeur : le transfert 1 Go
    chute anormalement ~2x sur ce portable, voir bench_d2d)."""
    bws = [r.get("bw_gbs") for r in curve if r.get("bw_gbs")]
    peak = max(bws) if bws else 0
    if not peak:
        return None
    for r in curve:
        if r.get("bw_gbs") and r["bw_gbs"] >= 0.9 * peak:
            return r["size"]
    return curve[-1]["size"]


def bench_dma_latency(cuda):
    """Latence DMA : petit H2D+D2H back-to-back, best of N."""
    size = 256
    h = (ctypes.c_ubyte * size)()
    d = cuda.alloc(size)
    best = 1e18
    for _ in range(20):
        ms = cuda.timed(lambda: (cuda.memcpy(d, ctypes.cast(h, ctypes.c_void_p), size, MEMCPY_KIND["H2D"]),
                                 cuda.memcpy(ctypes.cast(h, ctypes.c_void_p), d, size, MEMCPY_KIND["D2H"])))
        best = min(best, ms)
    cuda.free(d)
    return {"latency_ms": round(best, 4), "latency_us": round(best * 1000, 1),
            "roundtrip": "H2D(256B) + D2H(256B) back-to-back"}


def bench_overlap(cuda, size_h2d=64 << 20, size_memset=1 << 30):
    """Copy/compute overlap : memcpyAsync (s1) + memsetAsync (s2), temps réel vs somme."""
    s1, s2 = ctypes.c_void_p(), ctypes.c_void_p()
    cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(s1)))
    cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(s2)))
    d = cuda.alloc(size_memset)
    e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
    cuda.check(cuda.lib.cudaEventCreate(ctypes.byref(e0)))
    cuda.check(cuda.lib.cudaEventCreate(ctypes.byref(e1)))

    ms = ctypes.c_float(0)

    def _seq():
        cuda.check(cuda.lib.cudaMemsetAsync(d, 0, size_memset, s1))
        cuda.check(cuda.lib.cudaMemsetAsync(d, 0, size_memset, s2))

    cuda.check(cuda.lib.cudaEventRecord(e0, None))
    _seq()
    cuda.check(cuda.lib.cudaEventRecord(e1, None))
    cuda.check(cuda.lib.cudaEventSynchronize(e1))
    cuda.check(cuda.lib.cudaEventElapsedTime(ctypes.byref(ms), e0, e1))
    seq = ms.value

    cuda.check(cuda.lib.cudaEventRecord(e0, None))
    cuda.check(cuda.lib.cudaMemsetAsync(d, 0, size_memset, s1))
    cuda.check(cuda.lib.cudaMemsetAsync(d, 0, size_memset, s2))
    cuda.check(cuda.lib.cudaEventRecord(e1, None))
    cuda.check(cuda.lib.cudaEventSynchronize(e1))
    cuda.check(cuda.lib.cudaEventElapsedTime(ctypes.byref(ms), e0, e1))
    overl = ms.value

    cuda.free(d)
    return {"sequential_ms": round(seq, 3), "overlapped_ms": round(overl, 3),
            "speedup": round(seq / max(overl, 1e-9), 2)}


def bench_cublas(cuda, m=4096, n=4096, k=4096, reps=5):
    """GEMM FP32/FP16 via cublas64_12.dll (TFLOPS mesurés)."""
    out = {"fp32_tflops": None, "fp16_tflops": None, "error": None}
    dlls = _find_cuda_dll("cublas64")
    if not dlls:
        out["error"] = "cublas64_*.dll introuvable"
        return out
    try:
        blas = ctypes.CDLL(dlls[0])
    except OSError as e:
        out["error"] = str(e)
        return out
    blas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    blas.cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    if blas.cublasCreate_v2(ctypes.byref(handle)) != 0:
        out["error"] = "cublasCreate failed"
        return out

    import numpy as np
    flops = 2.0 * m * n * k

    def _gemm_f32():
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        c = np.empty((m, n), dtype=np.float32)
        da, db, dc = cuda.alloc(a.nbytes), cuda.alloc(b.nbytes), cuda.alloc(c.nbytes)
        cuda.memcpy(da, a.ctypes.data_as(ctypes.c_void_p), a.nbytes, MEMCPY_KIND["H2D"])
        cuda.memcpy(db, b.ctypes.data_as(ctypes.c_void_p), b.nbytes, MEMCPY_KIND["H2D"])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        blas.cublasSgemm_v2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_void_p, ctypes.c_int]
        blas.cublasSgemm_v2.restype = ctypes.c_int
        e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
        cuda.check(cuda.lib.cudaEventCreate(ctypes.byref(e0)))
        cuda.check(cuda.lib.cudaEventCreate(ctypes.byref(e1)))
        ms = ctypes.c_float(0)
        blas.cublasSgemm_v2(handle, 0, 0, m, n, k, ctypes.byref(alpha),
                            da, m, db, k, ctypes.byref(beta), dc, m)
        cuda.check(cuda.lib.cudaEventRecord(e0, None))
        for _ in range(reps):
            blas.cublasSgemm_v2(handle, 0, 0, m, n, k, ctypes.byref(alpha),
                                da, m, db, k, ctypes.byref(beta), dc, m)
        cuda.check(cuda.lib.cudaEventRecord(e1, None))
        cuda.check(cuda.lib.cudaEventSynchronize(e1))
        cuda.check(cuda.lib.cudaEventElapsedTime(ctypes.byref(ms), e0, e1))
        cuda.free(da)
        cuda.free(db)
        cuda.free(dc)
        return flops * reps / (ms.value / 1000) / 1e12

    try:
        out["fp32_tflops"] = round(_gemm_f32(), 2)
    except Exception as e:
        out["fp32_error"] = str(e)

    blas.cublasHgemm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int]
    blas.cublasHgemm.restype = ctypes.c_int

    def _gemm_f16():
        one = ctypes.c_uint16(0x3C00)
        a = np.ones((m, k), dtype=np.float16)
        b = np.ones((k, n), dtype=np.float16)
        c = np.empty((m, n), dtype=np.float16)
        da, db, dc = cuda.alloc(a.nbytes), cuda.alloc(b.nbytes), cuda.alloc(c.nbytes)
        cuda.memcpy(da, a.ctypes.data_as(ctypes.c_void_p), a.nbytes, MEMCPY_KIND["H2D"])
        cuda.memcpy(db, b.ctypes.data_as(ctypes.c_void_p), b.nbytes, MEMCPY_KIND["H2D"])
        e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
        cuda.check(cuda.lib.cudaEventCreate(ctypes.byref(e0)))
        cuda.check(cuda.lib.cudaEventCreate(ctypes.byref(e1)))
        ms = ctypes.c_float(0)
        blas.cublasHgemm(handle, 0, 0, m, n, k, ctypes.byref(one),
                            da, m, db, k, ctypes.byref(one), dc, m)
        cuda.check(cuda.lib.cudaEventRecord(e0, None))
        for _ in range(reps):
            blas.cublasHgemm(handle, 0, 0, m, n, k, ctypes.byref(one),
                             da, m, db, k, ctypes.byref(one), dc, m)
        cuda.check(cuda.lib.cudaEventRecord(e1, None))
        cuda.check(cuda.lib.cudaEventSynchronize(e1))
        cuda.check(cuda.lib.cudaEventElapsedTime(ctypes.byref(ms), e0, e1))
        cuda.free(da)
        cuda.free(db)
        cuda.free(dc)
        return flops * reps / (ms.value / 1000) / 1e12

    try:
        out["fp16_tflops"] = round(_gemm_f16(), 2)
    except Exception as e:
        out["fp16_error"] = str(e)
    return out


def bench_cpu_gemm(n=1024, reps=3):
    import numpy as np
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    c = a @ b
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return {"m": n, "gflops": round(2.0 * n ** 3 / best / 1e9, 1), "ms": round(best * 1000, 2)}


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------
def build_fingerprint(args):
    fp = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": {
            "os": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "cpu": probe_cpu(),
        "ram": probe_ram(),
        "gpu": probe_gpu_smi(),
        "npu": probe_npu(),
        "vram": None,
        "pcie": None,
        "dma": None,
        "compute": {"cpu": None, "gpu": None},
        "memory": None,
        "microbench": None,
    }

    fp["compute"]["cpu"] = bench_cpu_gemm()

    if args.no_cuda or not fp["gpu"]["present"]:
        fp["note"] = "CUDA désactivé/absent : vram/pcie/dma/gpu compute non mesurés"
        return fp

    dlls = _find_cuda_dll("cudart64")
    try:
        cuda = CudaRuntime(dlls)
    except Exception as e:
        fp["note"] = f"cudart indisponible : {e} — vram/pcie/dma non mesurés"
        return fp

    cuda.check(cuda.lib.cudaSetDevice(0))

    fp["vram"] = {
        "capacity_gb": round((fp["gpu"].get("vram_mib") or 0) / 1024.0, 1),
        "theoretical_peak_gbs": None,
        "measured_read_gbs": None,
        "measured_write_gbs": None,
        "measured_copy_gbs": None,
        "source": "measure",
        "note": "theoretical_peak_gbs=null (bus width non expose par nvidia-smi) ; "
                "measured_read_gbs=null tant que pas de kernel reduction (phase 2) ; "
                "measured_write_gbs=memset (moteur DMA) ; "
                "H2D/D2H refletent le chemin PCIe, PAS la bande passante VRAM",
    }

    try:
        d2d = bench_d2d(cuda, [64 << 20, 256 << 20, 1 << 30])
        ok_bw = [r["bw_gbs"] for r in d2d if r.get("bw_gbs")]
        # Le transfert 1 Go chute anormalement (~2x) : on garde le PIC de la courbe
        # comme bande passante représentative, pas la dernière valeur (1 Go).
        fp["vram"]["measured_copy_gbs"] = round(max(ok_bw), 2) if ok_bw else None
        fp["vram"]["d2d_curve"] = d2d
        fp["vram"]["measured_write_gbs"] = bench_memset_write(cuda)
    except Exception as e:
        fp["vram"]["error"] = str(e)

    sizes = args.sizes or DEFAULT_SIZES
    h2d = bench_transfer_curve(cuda, sizes, MEMCPY_KIND["H2D"])
    d2h = bench_transfer_curve(cuda, sizes, MEMCPY_KIND["D2H"])
    max_gen = fp["gpu"].get("pcie_max_gen") or 3
    cur_w = fp["gpu"].get("pcie_cur_width") or 16
    theo_cur = (2 ** max_gen) * (128.0 / 130.0) / 8.0 * cur_w  # Gen4 x8 = 15.754
    big_h2d = [r for r in h2d if r.get("bw_gbs")]
    big_d2h = [r for r in d2h if r.get("bw_gbs")]
    # PIC de la courbe (la valeur 1 Go chute anormalement sur ce portable).
    eff_h2d = round(max(r["bw_gbs"] for r in big_h2d), 2) if big_h2d else None
    eff_d2h = round(max(r["bw_gbs"] for r in big_d2h), 2) if big_d2h else None
    fp["pcie"] = {
        "generation": max_gen,
        "configured_width": cur_w,
        "configured_generation": max_gen,
        "theoretical_gbs": round(theo_cur, 3),
        "measured_h2d_gbs": eff_h2d,
        "measured_d2h_gbs": eff_d2h,
        "measured_gbs": eff_h2d,
        "efficiency": round((eff_h2d or 0) / theo_cur, 3) if eff_h2d else None,
        "note": "theoretical_gbs = Gen4 x8 apres overhead 128b/130b "
                "(2^gen GT/s/lane * lanes * 128/130 / 8). "
                "nvidia-smi peut rapporter Gen2 x8 au repos (idle) : "
                "la liaison monte en Gen4 sous charge. "
                "measure = 1GB H2D best-of-3.",
        "curve_h2d": h2d,
        "curve_d2h": d2h,
    }

    fp["dma"] = bench_dma_latency(cuda)
    fp["dma"]["bandwidth_gbs"] = fp["vram"].get("measured_copy_gbs")
    fp["dma"]["setup_latency_us"] = None
    fp["dma"]["completion_latency_us"] = None
    fp["dma"]["min_transfer"] = min(DEFAULT_SIZES)
    fp["dma"]["optimal_transfer"] = _optimal_transfer(h2d)
    fp["dma"]["note"] = ("T_dma = setup + bytes/bandwidth. Pour H2D/D2H le transfert "
                         "emprunte le chemin PCIe : ne PAS additionner T_dma_transfer "
                         "et T_pcie pour le meme transfert (double comptage). "
                         "latence 17us dominante pour <64KB, negligeable au-dela de 16MB.")
    try:
        fp["dma"]["overlap"] = bench_overlap(cuda)
    except Exception as e:
        fp["dma"]["overlap_error"] = str(e)

    fp["compute"]["gpu"] = bench_cublas(cuda)

    fp["microbench"] = {
        "sizes": SIZE_LABELS,
        "readme": "courbe transport H2D/D2H = (size, bw_gbs, ms) ; PCIe/VRAM = bande passante mesurée"
    }

    ram_bw = fp["ram"].get("measured_bw_gbs")
    vram_bw = (fp["vram"] or {}).get("measured_copy_gbs")
    pcie_h2d = (fp["pcie"] or {}).get("measured_h2d_gbs")
    fp["memory"] = {
        "ram_bw_gbs": ram_bw,
        "vram_bw_gbs": vram_bw,
        "pcie_h2d_gbs": pcie_h2d,
        "pcie_d2h_gbs": (fp["pcie"] or {}).get("measured_d2h_gbs"),
        "rule": "NE PAS sommer ram+vram+pcie : modéliser le critical path avec overlap",
    }
    fp["topology"] = {
        "path": "CPU RAM --PCIe Gen4 x8--> GPU VRAM",
        "dma_engine": "GPU (moteur copy/DMA)",
        "rule": "H2D/D2H empruntent le meme chemin PCIe : un transfert CPU<->GPU est compte "
                "UNE fois (T_pcie), le setup DMA en plus. T_total = T_compute + T_dequant "
                "+ T_dma_setup + T_transfer + T_sync.",
        "double_count_risk": "ne pas additionner T_dma_transfer et T_pcie pour le meme octet",
    }
    return fp


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="D2 HW PROBE — fingerprint matériel mesuré")
    ap.add_argument("--json", default=os.path.join(HERE, "hardware_fingerprint.json"))
    ap.add_argument("--no-cuda", action="store_true",
                    help="ne pas charger cudart (CPU/RAM/GPU nvidia-smi uniquement)")
    ap.add_argument("--sizes", type=str, default=None,
                    help="taille de transfert séparées par des virgules (k/m/g) pour la courbe PCIe")
    args = ap.parse_args()

    size_list = None
    if args.sizes:
        size_list = []
        mult = {"k": 1024, "m": 1 << 20, "g": 1 << 30, "b": 1}
        for tok in args.sizes.replace(" ", "").split(","):
            unit = tok[-1].lower() if tok[-1].lower() in mult else "b"
            num = tok[:-1] if unit != "b" else tok
            size_list.append(int(num) * mult[unit])
    args.sizes = size_list

    fp = build_fingerprint(args)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(fp, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in fp.items() if k != "microbench"},
                     ensure_ascii=False, indent=2)[:4000])
    print(f"\n[+] fingerprint : {args.json}")


if __name__ == "__main__":
    main()