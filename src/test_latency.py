import json, time, urllib.request

URL = "http://127.0.0.1:8000/predict"
TEXT = "The government announced a new policy affecting millions of citizens."


def call():
    req = urllib.request.Request(
        URL, data=json.dumps({"text": TEXT}).encode(),
        headers={"Content-Type": "application/json"},
    )
    s = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        body = json.loads(r.read())
    return (time.perf_counter() - s) * 1000, body["latency_ms"]


for _ in range(3):
    call()  # warm-up
res = [call() for _ in range(50)]
total = sorted(r[0] for r in res)
model_only = sorted(r[1] for r in res)
print(f"End-to-end: avg {sum(total)/50:.1f} ms | p95 {total[47]:.1f} ms")
print(f"Model only: avg {sum(model_only)/50:.1f} ms | p95 {model_only[47]:.1f} ms")