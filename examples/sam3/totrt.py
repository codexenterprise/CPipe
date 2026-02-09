from cpipe.module.model.yolov10 import YOLOv10
max_batch_size = 1
# MODEL_PATH = "/mnt/d/models/sam3/decoder-fp16.onnx"
# YOLOv10.onnx2tensorrt(MODEL_PATH, input_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2", "prompt_features", "prompt_mask"],
#                              min_shapes=[(1, 256, 288, 288), (1, 256, 144, 144), (1, 256, 72, 72), (1, 256, 72, 72), (1, 1, 256), (1,1)],
#                               opt_shapes=[(1, 256, 288, 288), (1, 256, 144, 144), (1, 256, 72, 72), (1, 256, 72, 72), (1, 33, 256), (1,33)],
#                                max_shapes=[(max_batch_size, 256, 288, 288), (max_batch_size, 256, 144, 144), (max_batch_size, 256, 72, 72), (max_batch_size, 256, 72, 72), (max_batch_size, 60, 256), (max_batch_size,60)]
#                                )

# MODEL_PATH = "/mnt/d/models/sam3/text-encoder-fp16.onnx"
# YOLOv10.onnx2tensorrt(MODEL_PATH, input_names=["input_ids", "attention_mask"],
#                              min_shapes=[(1, 32), (1, 32)],
#                               opt_shapes=[(1, 32), (1, 32)],
#                                max_shapes=[(max_batch_size, 32), (max_batch_size, 32)],
#                                fp16_mode=False
#                                )

MODEL_PATH = "/mnt/d/models/sam3/geometry-encoder.onnx"
YOLOv10.onnx2tensorrt(MODEL_PATH, input_names=["input_boxes", "input_boxes_labels", "fpn_feat_2", "fpn_pos_2"],
                             min_shapes=[(1, 1, 4), (1, 1), (1, 256, 72, 72), (1, 256, 72, 72)],
                              opt_shapes=[(1, 8, 4), (1, 8), (1, 256, 72, 72), (1, 256, 72, 72)],
                               max_shapes=[(max_batch_size, 20, 4), (max_batch_size, 20), (max_batch_size, 256, 72, 72), (max_batch_size, 256, 72, 72)],
                               fp16_mode=True,
                               int8_mode=True
                               )

# MODEL_PATH = "/mnt/d/models/sam3/vision-encoder-fp16.onnx"
# YOLOv10.onnx2tensorrt(MODEL_PATH, input_names=["images"],
#                              min_shapes=[(1, 3, 1008, 1008)],
#                               opt_shapes=[(1, 3, 1008, 1008)],
#                                max_shapes=[(max_batch_size, 3, 1008, 1008)],
                               
#                                )

