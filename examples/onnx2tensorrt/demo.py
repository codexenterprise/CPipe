from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.model.shufflenet import ShuffleNet
from cpipe.module.model.adaface import Adaface
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.model.yolov11 import YOLOv11
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.model.movenet import MoveNet, MoveNetPersonPose
from cpipe.module.model.mmsegmentation import MMSemanticSegmentation

# need install tensorrt, cuda, cuDNN, torch2trt
MODEL_PATH = "/home/zhouhe/workspace/cpipe2.0/__OTHERS__/demo_person/yolov10n.onnx.codex"
# MODEL_PATH = "/home/zhouhe/workspace/cpipe2.0/__OTHERS__/demo_person/movenet_person_pose.onnx.cpipe"

# with dynamic batch size
YOLOv7.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=640, input_width=640)
YOLOv10.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=640, input_width=640)
YOLOv11.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=640, input_width=640)

YOLOv8obb.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=640, input_width=640)

Adaface.onnx2tensorrt(MODEL_PATH, max_batch_size=64, input_height=112, input_width=112, fp16_mode=True, int8_mode=False)

Retinaface.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=416, input_width=416)

ShuffleNet.onnx2tensorrt(MODEL_PATH, max_batch_size=64, input_height=224, input_width=224, fp16_mode=True, int8_mode=False)

MoveNet.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=256, input_width=256)

MoveNetPersonPose.onnx2tensorrt(MODEL_PATH, input_height=256, input_width=256)

MMSemanticSegmentation.onnx2tensorrt(MODEL_PATH, max_batch_size=16, input_height=512, input_width=512)
