# -*- coding: utf-8 -*-
"""Scores both engines of the running web server against the CFP filing on the cached Devoxx talks.
   usage: python3 java/tools/eval/devoxx_eval.py [count]   (server on localhost:8000)"""
import json, urllib.request, collections, sys
p = json.load(open("presets/devoxx_cfp.json"))
base = {k: v for k, v in p.items() if k != "samples"}
keys = ["track", "audience_level"]
hits = {"parallel": collections.Counter(), "baseline": collections.Counter()}
def race(body):
    req = urllib.request.Request("http://localhost:8000/api/run/race", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = {}; ev = None
    for line in urllib.request.urlopen(req):
        line = line.decode().rstrip("\n")
        if line.startswith("event: "): ev = line[7:]
        elif line.startswith("data: ") and ev in ("parallel", "naive"): out[ev] = json.loads(line[6:])
    return out
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(p["samples"])
for s in p["samples"][:n]:
    body = dict(base); body["context"] = s["context"]
    r = race(body)
    par = {f["name"]: f["value"] for f in r["parallel"]["fields"]}
    nj = r["naive"].get("json") or {}
    for k in keys:
        hits["parallel"][k] += par.get(k) == s["expected"][k]
        hits["baseline"][k] += str(nj.get(k)) == s["expected"][k]
model = json.load(urllib.request.urlopen("http://localhost:8000/api/status"))["model"]
print(f"{n} talks, model {model}")
for k in keys: print(f"  {k:16} parallel {hits['parallel'][k]:3}/{n}   baseline {hits['baseline'][k]:3}/{n}")
