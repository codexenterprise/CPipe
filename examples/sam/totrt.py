from cpipe.module.model.dino import DinoEmbedding
from cpipe.module.model.fastsam import FastSAM

# need install tensorrt, cuda, cuDNN, torch2trt
# MODEL_PATH = "../../src/model_files/dino_embedding.onnx"

# MODEL_PATH = "/home/zhouhe/workspace/cpipe2.0/__OTHERS__/demo_person/movenet_person_pose.onnx.cpipe"

# with dynamic batch size
# DinoEmbedding.onnx2tensorrt(MODEL_PATH, input_names=["input"], min_shapes=[(1, 3, 224, 224)], opt_shapes=[(1, 3, 224, 224)], max_shapes=[(32, 3, 224, 224)])

MODEL_PATH = "./src/model_files/FastSAM-x.onnx"
FastSAM.onnx2tensorrt(MODEL_PATH, input_names=["images"], min_shapes=[(1, 3, 576, 1024)], opt_shapes=[(1, 3, 576, 1024)], max_shapes=[(1, 3, 576, 1024)])