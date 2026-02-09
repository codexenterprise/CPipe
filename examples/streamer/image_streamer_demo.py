import multiprocessing
import os
import time

import cv2

from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import ImageStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


def thread_func(queue, img_path):
    file_path = os.listdir(img_path)
    while True:
        for one in file_path:
            print(0, one)
            queue.put({"images_path": [os.path.join(img_path, one)], "images_array": [], "images_info": [{"1": 1}]})  # images_path and images_array are mutually exclusive, cannot exist at the same time
            time.sleep(1)
        # for one in file_path: # this way is slow
        #     print(1, one)
        #     queue.put({"images_path": [], "images_array": [cv2.imread(os.path.join(img_path, one))], "images_info": [{"1": 1}]})
        #     time.sleep(1)


if __name__ == "__main__":
    streamer_nodes = []
    streams_rtsp = []
    queue = multiprocessing.Queue(10)
    # stream support queue(multiprocessing.Queue) mode and file(folder or image file path) mode
    stream = ImageStreamer(node_name="stream", stream=queue, output_wh=(1920, 1080), padding=False, play_interval=0.04, once_mode=False, ground_image_path="./data/img.png")

    p = multiprocessing.Process(target=thread_func, args=(queue, "/mnt/d/to_upload"))
    p.start()

    detect = YOLOv10(node_name="YOLOv10",
                     model_path="src/model_files/yolov10n.onnx",
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

    # tianping_pose = MoveNetPersonPose(
    #                                   model_path="./movenet_person_pose.onnx.codex",
    #                                   input_size=(3, 256, 256),
    #                                   num_joints=17,
    #                                   secondary_class_names=["person"]
    #                                   )

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [detect, cpipeinsight]

    #  start all initialized nodes
    Node.launch(check_node=True, auto_restart=False)
