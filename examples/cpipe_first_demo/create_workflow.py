from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from examples.personnel_intrusion.custom_node import PersonnelIntrusion

if __name__ == "__main__":
    # stream node
    stream1 = VideoStreamer(stream="rtmp://192.168.10.7:1935/live/7777",
                            process_frame_interval=1, # Take 1 frame for processing every process_frame_interval frames.
                            )

    # algorithm node
    detect = YOLOv10(
                    model_path="model_files/yolov10x_batch1.engine", # The path to the model file.
                    # if class_names is not provided, the class names will be read from the label file of the model(model_path.split(".")[0] + ".txt").
                    # class_names=['person', ...], 
                    valid_class_names=["person"], # The valid class names of the model. If valid_class_names is not provided, the model will output all the class names.
                    )

    # logic node
    personnel_intrusion = PersonnelIntrusion()

    # insight node
    cpipeinsight = CPipeInsight()

    # link all nodes
    stream1 += [detect, personnel_intrusion, cpipeinsight]

    # launch the workflow
    Node.launch(check_node=True, auto_restart=False)
