# onnx -> har 

import onnx 
from hailo_sdk_client import ClientRunner 

onnx_model_name = "yolov11"
onnx_path = "./model/best.onnx"
chosen_hw_arch = "hailo8"

# get onnx model input and output names
model = onnx.load(onnx_path)
print("=== ONNX INPUTS ===")
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  name: {inp.name},  shape: {dims}")

# --- 출력 이름 ---
print("=== ONNX OUTPUTS ===")
for out in model.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  name: {out.name},  shape: {dims}")

runner = ClientRunner(hw_arch = chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
    onnx_path, 
    onnx_model_name,
    start_node_names = ["images"],
    end_node_names = [        
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv",
        ],
    net_input_shapes = {"images": [1,3,640,640]},
    )

# save 
hailo_model_har_name = f"{onnx_model_name}_hailo_model.har"
runner.save_har(hailo_model_har_name)


