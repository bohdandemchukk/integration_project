"""python scripts/benchmark.py http://localhost:8000 samples/yes.wav 200"""
import sys
import time

import numpy as np
import requests

url, wav, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = open(wav, "rb").read()
s = requests.Session()


def call():
    t0 = time.perf_counter()
    r = s.post(f"{url}/predict", files={"file": ("a.wav", data, "audio/wav")})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000, r.json()["latency_ms"]


for _ in range(10):  # прогрів
    call()

client, server = map(np.array, zip(*(call() for _ in range(n))))
for name, a in (("server", server), ("client", client)):
    print(f"{name}: mean={a.mean():.2f}  p50={np.percentile(a, 50):.2f}  "
          f"p95={np.percentile(a, 95):.2f} ms  (n={n})")