import json, statistics, time, urllib.request
KEY="REDACTED_SET_SGLANG_API_KEY"
BASE="http://10.100.8.1:8940"
def gen(ids,mn):
    t0=time.perf_counter()
    req=urllib.request.Request(BASE+"/generate",data=json.dumps({"input_ids":ids,"sampling_params":{"max_new_tokens":mn,"temperature":0.0}}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    with urllib.request.urlopen(req,timeout=1800) as r: m=json.loads(r.read().decode())["meta_info"]
    return time.perf_counter()-t0, m
print("L | TTFT(ms, mn=1) | decode ms/token = (t33-t1)/32")
for L in (4096, 32768, 100000):
    ids=[100+(i%50) for i in range(L)]
    gen(ids,2)                      # warm
    t1s=[];t33s=[]
    for _ in range(5):
        t1,_=gen(ids,1); t33,_=gen(ids,33)
        t1s.append(t1); t33s.append(t33)
    ttft=statistics.median(t1s)*1000
    dtok=(statistics.median(t33s)-statistics.median(t1s))/32*1000
    print(f"L={L:6d} TTFT={ttft:7.1f}ms  decode={dtok:6.2f} ms/token")
