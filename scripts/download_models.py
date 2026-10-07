"""Downloads the vehicle detector. fast-alpr fetches its own plate models on first use.

Check the licence of every weight file before commercial use (YOLOX code is Apache-2.0;
weights are COCO-trained -- confirm the terms you are comfortable with).
If the URL ever changes, download yolox_nano.onnx from the YOLOX GitHub releases page by hand
and put it in models/.
"""
import os
import urllib.request

URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
DEST = os.path.join("models", "yolox_nano.onnx")

os.makedirs("models", exist_ok=True)
if os.path.exists(DEST):
    print("already present:", DEST)
else:
    print("downloading", URL)
    urllib.request.urlretrieve(URL, DEST)
    print("saved", DEST, os.path.getsize(DEST) // 1024, "KB")
