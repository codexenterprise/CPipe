import os
import pickle

import numpy as np

from cpipe.module.model.facematching import FaceLibrary
from cpipe.module.model.facerecognition import FaceRecognition
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.model.yolov26 import YOLO26
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    streamer_nodes = []
    streams_rtsp = []
    # 
    stream = VideoStreamer(node_name="stream", stream="rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101", process_frame_interval=1)

    detect = YOLO26(
                     model_path="model_files/yolo26n.engine",
                     valid_class_names=["person"],
                     )

    rf = Retinaface(
        model_path="model_files/retinaface_416x416-det_10g_batch.engine",
        secondary_class_names=["person"],
    )

    # face_embeddings = []
    # face_names = []
    # embedding_files_path = "./face_embeddings"
    # for one in os.listdir(embedding_files_path):
    #     with open(os.path.join(embedding_files_path, one), "rb") as f:
    #         face_embeddings.append(pickle.load(f))
    #         face_names.append(one.split(".")[0][5:])
    # face_embeddings = np.array(face_embeddings)
    # fl = FaceLibrary(face_embeddings, face_names)

    fr = FaceRecognition(
        model_path="model_files/adaface.engine",
        face_images_path="./face_images",
        # face_quality_model_path="../../src/model_files/face_quality_batch64_GPU3070.engine",
        secondary_class_names=["person"],
        # faces_library=fl,
        matching_score_thresh=0.1
    )

    cpipeinsight = CPipeInsight(http_insight=True, show_key_points=False, save_video=False)

    stream += [detect, rf, fr, cpipeinsight]

    #  launch all initialized nodes
    Node.launch(check_node=True, auto_restart=False)
