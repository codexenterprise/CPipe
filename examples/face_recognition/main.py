import os
import pickle

import numpy as np

from cpipe.module.model.facematching import FaceLibrary
from cpipe.module.model.facerecognition import FaceRecognition
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    streamer_nodes = []
    streams_rtsp = []
    stream = VideoStreamer(node_name="stream", stream="rtmp://192.168.10.7:1935/live/7777", process_frame_interval=1, once_mode=True)

    detect = YOLOv7(node_name="YOLOv7",
                     model_path="src/model_files/yolov7-tiny_office_0.65_0.45.engine",
                     input_size=(3, 640, 640),
                     class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                                  'hair drier', 'toothbrush'],
                     max_batch_size=1,
                     valid_class_names=["person"],
                     save_top_n_objects=32,
                     area_flag=True
                     )

    rf = Retinaface(
        node_name="retinaface",
        model_path="src/model_files/416x416-det_10g_batch.engine",
        input_size=(3, 416, 416),
        class_names=["face"],
        max_batch_size=64,
        secondary_class_names=["person"],
    )

    face_embeddings = []
    face_names = []
    embedding_files_path = "./face_embeddings"
    for one in os.listdir(embedding_files_path):
        with open(os.path.join(embedding_files_path, one), "rb") as f:
            face_embeddings.append(pickle.load(f))
            face_names.append(one.split(".")[0][5:])
    face_embeddings = np.array(face_embeddings)
    fl = FaceLibrary(face_embeddings, face_names)

    fr = FaceRecognition(
        node_name="adaface",
        model_path="src/model_files/adaface_ir101_webface12m.engine",
        input_size=(3, 112, 112),
        max_batch_size=64,
        # face_quality_model_path="../../src/model_files/face_quality_batch64_GPU3070.engine",
        secondary_class_names=["person"],
        faces_library=fl,
        matching_score_thresh=0.1
    )

    cpipeinsight = CPipeInsight(http_insight=True, show_key_points=False, save_video=False)

    stream += [detect, rf, fr, cpipeinsight]

    #  launch all initialized nodes
    Node.launch(check_node=True, auto_restart=False)
