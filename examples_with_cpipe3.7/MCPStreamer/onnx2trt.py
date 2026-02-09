from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.model.adaface import Adaface
from cpipe.module.model.retinaface import Retinaface

# need install tensorrt, cuda, cuDNN, torch2trt
MODEL_PATH = "/home/zhouhe/workspace/cpipe2.0/src/model_files/416x416-det_10g_batch.onnx.cpipe"
Adaface.onnx2tensorrt(MODEL_PATH, max_batch_size=64, input_height=112, input_width=112, fp16_mode=True, int8_mode=False)
MODEL_PATH = "/home/zhouhe/workspace/cpipe2.0/src/model_files/adaface_ir101_webface12m.onnx.cpipe"
Retinaface.onnx2tensorrt(MODEL_PATH, max_batch_size=64, input_height=416, input_width=416)
