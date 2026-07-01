import os
import pickle
import numpy as np

from cpipe.module.model.facematching import FaceLibrary
from cpipe.module.model.facerecognition import FaceRecognition
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

def gstreamer_pipeline(url, codec):
    pipeline = (
        f"rtspsrc location={url} latency=0 ! "
        f"rtp{codec}depay ! {codec}parse ! "
        "mppvideodec format=BGR ! "
        "appsink drop=1 max-buffers=2 sync=false"
    )
    return pipeline

if __name__ == "__main__":
    streamer_nodes = []
    streams_rtsp = []
    # stream = VideoStreamer(node_name="stream", stream="rtmp://192.168.8.122:1935/live/7777", process_frame_interval=24, once_mode=True)
    
    # hardware acceleration with GSTREAMER 
    stream = VideoStreamer(stream=gstreamer_pipeline("rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101", "h264"), process_frame_interval=1, capture_mode="GSTREAMER")#
    
    
    #            device: The device of the model. eg: rknn:0
    #                     NPU_CORE_AUTO  = 0                                   # default, run on NPU core randomly.
    #                     NPU_CORE_0     = 1                                   # run on NPU core 0.
    #                     NPU_CORE_1     = 2                                   # run on NPU core 1.
    #                     NPU_CORE_2     = 4                                   # run on NPU core 2.
    #                     NPU_CORE_0_1   = 3                                   # run on NPU core 1 and core 2.
    #                     NPU_CORE_0_1_2 = 7                                   # run on NPU core 1 and core 2 and core 3.
    #                     NPU_CORE_ALL   = 0xffff                              # run on all NPU cores.
    # time:  0.07101058959960938
    detect = YOLOv7(
                    model_path="/home/rockchip/zh/cpipe2.0/examples/face_recognition/model/yolov7-tiny_batch_std255.rknn", 
                    max_batch_size=1,  # "rknn model only support batch size 1"
                    valid_class_names=["person"],
                    device="rknn:0",
                    anchor=np.array([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]).reshape(3, -1, 2).tolist()
                    )
    
    # detect = YOLOv10(
    #             model_path="./model_files/yolo26n-rk3588.rknn",
    #             valid_class_names=["person"], 
    #             device="rknn:4",
    # )

    tk = Tracker(node_name="tracker1", tacker_type='ocsort', scale_ratio=4.0, secondary_class_names=["person"], dump_images=True)

    # time:  0.29663944244384766
    rf = Retinaface(
        model_path="/home/rockchip/zh/cpipe2.0/examples/face_recognition/model/416x416-det_10g_batch_std255.rknn",
        input_size=(3, 416, 416),
        class_names=["face"],
        max_batch_size=16,
        secondary_class_names=["person"],
        device="rknn:0",
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

    # time:  0.2530810832977295
    fr = FaceRecognition(
        node_name="adaface",
        model_path="/home/rockchip/zh/cpipe2.0/examples/face_recognition/model/adaface_ir101_webface12m.rknn",
        input_size=(3, 112, 112),
        max_batch_size=16,
        secondary_class_names=["person"],
        faces_library=fl,
        matching_score_thresh=0.1,
        device="rknn:0",
    )

    cpipeinsight = CPipeInsight(http_insight=True, show_key_points=False, save_video=True)

    stream += [detect, tk, rf, fr, cpipeinsight]

    Node.launch(check_node=True, auto_restart=False)
