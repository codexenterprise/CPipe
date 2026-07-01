from cpipe.module.insight import CPipeInsight
from cpipe.module.model.vlm import VLM
from cpipe.module.model.yolov26 import YOLO26
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer
from examples.vlm_node.my_logic import my_node_with_to_vlm, my_node_report

if __name__ == "__main__":

    streamer = VideoStreamer(node_name="video_streamer", stream="demo.mp4", once_mode=True, process_frame_interval=24)

    ce_det = YOLO26(model_path="model_files/yolo26x.engine", valid_class_names=["person"])
    

    # save the result to the local file
    to_vlm = my_node_with_to_vlm(node_name="to_vlm")


    vlm = VLM(
        # set the max pixels of the image, if the image is larger than the max pixels, the image will be scaled down. That can raise the performance.
        image_max_pixels=int(256*256),
        base_url="http://192.168.10.77:8000/v1",
        api_key="ezajbsuwuwhvgaax.....", # if you want to use your own API key, you can set it here
        model_name="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
        max_tokens=50000, temperature=0.2, enable_thinking=False, timeout=120,
        # system prompt is optional
    )

    my_node_report = my_node_report(node_name="my_node_report")

    insight = CPipeInsight()
    streamer += [ce_det, to_vlm, vlm, my_node_report, insight]

    Node.launch()