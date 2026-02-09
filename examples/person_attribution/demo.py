from cv2 import Tracker
from cpipe.module.model.pphgnet import PPHGNet
# from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="rtsp://admin:tp123456@192.168.10.199:554/Streaming/Channels/101", process_frame_interval=1, once_mode=True)
    # detect = YOLOv10(
    #                 model_path="model_files/yolov10n_batch1.engine",
    #                 class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #                             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #                             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #                             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #                             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #                             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #                             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    #                             'hair drier', 'toothbrush'],
    #                 valid_class_names=["person"],
    #                 save_top_n_objects=32,
    #                 area_flag=False
    #                 )
    detect = YOLOv7(
                    model_path="model_files/mot_ppyoloe_l_36e_pipeline.onnx",
                    class_names=['person'],
                    valid_class_names=["person"],
                    save_top_n_objects=32,
                    area_flag=False
                    )
    track = Tracker(
        model_path="./model_files/bytetrack_s_mot17.onnx",
        secondary_class_names=["person"],
        max_batch_size=1,
        valid_class_names=["person"],
        save_top_n_objects=32,
    )
    person_attribution = PPHGNet(model_path="./model_files/PPHGNet_small_person_attribute_954_infer.onnx", secondary_class_names=["person"])

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2)

    stream1 += [detect, person_attribution, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, agent=False)
