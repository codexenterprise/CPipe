from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":
    # stream = VideoStreamer("ss", "rtsp://admin:tp123456@192.168.8.199:554/Streaming/Channels/101", 3, 1)
    stream = VideoStreamer("ss", "rtmp://192.168.8.7:1935/live/7777", 3, 1)
    detect = YOLOv10("D1",
                    "src/model_files/yolov10n.onnx",
                    3,
                    (3, 640, 640),
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
    # ynm = YOLOv10("D2",
    #                  "../../src/model_files/yolov10n.onnx",
    #                  3,
    #                  (3, 640, 640),
    #                  class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #                               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #                               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #                               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #                               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #                               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #                               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    #                               'hair drier', 'toothbrush'],
    #                  max_batch_size=1,
    #                  valid_class_names=["person"],
    #                  save_top_n_objects=32,
    #                  area_flag=True
    #                  )
    #------------------method 1: set mask for other nodes through streamer node--------------------------------
    # set one polygon mask(for YOLOv10 Node)
    stream.cmask.add_polygon("D1", "area", [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    # set one polygon(for YOLOv10 Node), but not provide polygon coordinates, draw on web
    stream.cmask.add_polygon("D1", "__area1__") # this is a required mask

    # set one line mask(for YOLOv10 Node)
    stream.cmask.add_line("D1", "line", [[0.3, 0.3], [0.4, 0.4]])
    # set one line(for YOLOv10 Node), but not provide line coordinates, draw on web
    stream.cmask.add_line("D1", "__line1__") # this is a required mask


    # #------------------method 2: set mask for algorithm(or custom node) node--------------------------------
    # # preset one polygon mask(for YOLOv10 Node), required=True, the mask name will be changed to __{mask_name}__
    # detect.preset_mask("p1", "polygons", required=True)
    # # preset one polygon mask(for YOLOv10 Node), required=False, the mask name will be changed to __{mask_name}__
    # detect.preset_mask("p2", "polygons")
    # # preset one line mask(for YOLOv10 Node), required=True, the mask name will be changed to __{mask_name}__
    # detect.preset_mask("l1", "lines", required=True)
    # # preset one line mask(for YOLOv10 Node), required=False
    # detect.preset_mask("l2", "lines")

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2)



    stream += [detect, cpipeinsight]

    # launch all initialized nodes
    Node.launch(check_node=True, auto_restart=False)
