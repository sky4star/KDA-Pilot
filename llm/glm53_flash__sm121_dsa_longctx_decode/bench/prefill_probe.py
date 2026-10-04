#!/usr/bin/env python3
"""Cold prefill / TTFT probe. Unique prompt per run (no prefix-cache hit)."""
import json, statistics, sys, time, urllib.request
KEY="REDACTED_SET_SGLANG_API_KEY"
BASE="http://10.100.8.1:8940"
_SALT=0
def ids_for(L):
    global _SALT; _SALT+=1; s=_SALT
    return [100+((i*7 + s*1013 + i//89 + (i*i)%7) % 50) for i in range(L)]
def gen(ids,mn):
    t0=time.perf_counter()
    req=urllib.request.Request(BASE+"/generate",data=json.dumps({"input_ids":ids,"sampling_params":{"max_new_tokens":mn,"temperature":0.0}}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    with urllib.request.urlopen(req,timeout=3600) as r: m=json.loads(r.read().decode())["meta_info"]
    return time.perf_counter()-t0, m
def main():
    runs=int(sys.argv[1]) if len(sys.argv)>1 else 3
    print("L | TTFT s (cold prefill) | tokens/s | cached")
    for L in (8192,32768,100000):
        gen(ids_for(min(L,2048)),1)   # warm JIT
        ws=[]
        for _ in range(runs):
            w,m=gen(ids_for(L),1); ws.append((w,m.get("cached_tokens")))
        w=statistics.median(x[0] for x in ws)
        print(f"L={L:6d} TTFT={w:7.2f}s ({L/w:8.1f} tok/s) cached={[x[1] for x in ws]}")
if __name__=="__main__": main()
