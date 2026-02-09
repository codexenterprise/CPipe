from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.model.shufflenet import ShuffleNet
from cpipe.module.model.adaface import Adaface
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.model.yolov11 import YOLOv11
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.model.movenet import MoveNet, MoveNetPersonPose
from cpipe.module.model.mmsegmentation import MMSemanticSegmentation

# 初始化TensorRT插件
# need install tensorrt, cuda, cuDNN, torch2trt
# MODEL_PATH = "model_files/416x416-det_10g_batch.onnx"
MODEL_PATH = "/mnt/d/kunshi2025.4.22.onnx"
# MODEL_PATH = "/home/zhouhe/workspace/cpipe2.0/__OTHERS__/demo_person/movenet_person_pose.onnx.cpipe"

# with dynamic batch size
# YOLOv10.onnx2tensorrt(MODEL_PATH, input_names=["tokens", "style", "speed"], min_shapes=[(1, 128), (1, 256), 1], opt_shapes=[(1, 128), (1, 256), 1], max_shapes=[(1, 512), (1, 256), 1])

# YOLOv10.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=640, input_width=640)

YOLOv11.onnx2tensorrt(MODEL_PATH, input_names=["images"], min_shapes=[(1, 3, 640, 640)], opt_shapes=[(1, 3, 640, 640)], max_shapes=[(1, 3, 640, 640)])

# YOLOv8obb.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=640, input_width=640)

# Adaface.onnx2tensorrt(MODEL_PATH, max_batch_size=64, input_height=112, input_width=112, fp16_mode=True, int8_mode=False)

# Retinaface.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=416, input_width=416)

# ShuffleNet.onnx2tensorrt(MODEL_PATH, max_batch_size=64, input_height=224, input_width=224, fp16_mode=True, int8_mode=False)

# MoveNet.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=256, input_width=256)

# MoveNetPersonPose.onnx2tensorrt(MODEL_PATH, input_height=256, input_width=256)

# MMSemanticSegmentation.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=512, input_width=512)
