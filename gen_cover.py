"""用 Agnes Image API 生成掘金封面（192x128 比例，3:2 出图后裁切）。"""
from __future__ import annotations

import base64
import json
import urllib.request

API = "https://api.agnes-ai.cn/v1/images/generations"
KEY = "sk-ssUmHcQBmJub2fEetYlE0snnSnMfQvtNnNa1ARHZ4FTOas2V"

PROMPT = (
    "Flat modern tech infographic cover art, dark navy blue background, "
    "three glowing connected nodes forming a horizontal data pipeline: "
    "left is a stack of 3 documents with soft cyan glow, "
    "middle is a sphere of small vector dots with purple glow, "
    "right is a friendly chatbot head with green glow, "
    "connected by clean thin white lines with subtle gradient. "
    "Minimalist, professional, no text, subtle vignette, "
    "3:2 aspect ratio, high quality digital illustration for a tech blog cover"
)

payload = json.dumps(
    {
        "model": "agnes-image-2.5-flash",
        "prompt": PROMPT,
        "n": 1,
        "size": "1536x1024",
    }
).encode()

req = urllib.request.Request(
    API,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    },
    method="POST",
)

with urllib.request.urlopen(req, timeout=90) as resp:
    data = json.loads(resp.read())

print("RAW RESPONSE KEYS:", list(data.keys()) if isinstance(data, dict) else type(data))
print("RAW (trunc):", json.dumps(data)[:800])
if "error" in data:
    print("API ERROR:", data["error"])
    raise SystemExit(1)

img = data["data"][0]
out = r"E:\rag-template\cover.png"
if img.get("b64_json"):
    raw = base64.b64decode(img["b64_json"])
    with open(out, "wb") as f:
        f.write(raw)
    print("saved b64 ->", out)
elif img.get("url"):
    req2 = urllib.request.Request(img["url"], method="GET")
    with urllib.request.urlopen(req2, timeout=60) as r2:
        with open(out, "wb") as f:
            f.write(r2.read())
    print("downloaded url ->", out, "size=", __import__("os").path.getsize(out))
else:
    print("no image data in response:", json.dumps(data)[:500])
