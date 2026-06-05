# har -> har_optimized -> hef
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import random
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

# ==================== 설정 ====================
HAR_PATH    = "./yolov11_hailo_model.har"
CALIB_DIR   = "./calib_images"
INPUT_SIZE  = 640
TEST_MODE   = False           
N_CALIB     = 256 if TEST_MODE else 1024

# ==================== NMS config JSON 생성 ====================
nms_config = {
    "nms_scores_th": 0.3,
    "nms_iou_th": 0.45,
    "image_dims": [640, 640],
    "max_proposals_per_class": 100,
    "classes": 1,
    "regression_length": 16,
    "background_removal": False,
    "bbox_decoders": [
        {"name": "bbox_decoder_8",  "stride": 8,  "reg_layer": "conv51", "cls_layer": "conv54"},
        {"name": "bbox_decoder_16", "stride": 16, "reg_layer": "conv62", "cls_layer": "conv65"},
        {"name": "bbox_decoder_32", "stride": 32, "reg_layer": "conv77", "cls_layer": "conv80"},
    ],
}
with open("nms_config.json", "w") as f:
    json.dump(nms_config, f, indent=2)
print("nms_config.json 생성됨")

# ==================== letterbox 전처리 ====================
def letterbox(img, size=640, pad=(114, 114, 114)):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    new = Image.new("RGB", (size, size), pad)
    new.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return new

# ==================== calibration 데이터 준비 ====================
files = [f for f in os.listdir(CALIB_DIR)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.seed(42)
random.shuffle(files)
files = files[:N_CALIB]

imgs = []
for f in files:
    img = Image.open(os.path.join(CALIB_DIR, f)).convert("RGB")
    img = letterbox(img, INPUT_SIZE)
    imgs.append(np.array(img, dtype=np.float32))   # 0~255 raw

calib = np.stack(imgs)
print(f"calib shape: {calib.shape}  (이미지 {len(files)}장)")

# ==================== HAR 로드 ====================
runner = ClientRunner(har=HAR_PATH)

# ==================== 모델 스크립트 ====================
model_script = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
nms_postprocess(config_path="nms_config.json", meta_arch=yolov8, engine=cpu)
"""
runner.load_model_script(model_script)

# ==================== Optimize ====================
print("=== Optimize 시작 (CPU라 시간 좀 걸림) ===")
runner.optimize(calib)
runner.save_har("./yolov11_quantized.har")
print("=== Optimize 완료 ===")

# ==================== Compile ====================
print("=== Compile 시작 ===")
hef = runner.compile()
with open("yolov11.hef", "wb") as f:
    f.write(hef)
print("=== HEF 생성 완료: yolov11.hef ===")