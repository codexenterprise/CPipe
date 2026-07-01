from cpipe.module.insight import CPipeInsight
from cpipe.module.model.vlm import VLM
from cpipe.module.model.yolov26 import YOLO26
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer
from examples.vlm_node.my_logic import my_node

if __name__ == "__main__":

    streamer = VideoStreamer(node_name="video_streamer", stream="demo.mp4", once_mode=True, process_frame_interval=24)

    ce_det = YOLO26(model_path="model_files/yolo26x.engine", valid_class_names=["person"])
    vlm = VLM(
        # set the max pixels of the image, if the image is larger than the max pixels, the image will be scaled down. That can raise the performance.
        image_max_pixels=int(256*256),
        # must to set secondary_class_names
        secondary_class_names=["person"],

        base_url="http://192.168.10.77:8000/v1",
        api_key="ezajbsuwuwhvgaax.....", # if you want to use your own API key, you can set it here
        model_name="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",

        max_tokens=50000, temperature=0.2, enable_thinking=False, timeout=120,
        # system prompt is optional
        system_prompt=(
            "你是工业安全合规检查助手。只输出符合给定 JSON Schema 的 JSON，"
            "字段全部使用布尔字面量（true / false），不要输出解释、注释或多余文本。"
        ),
        # question prompt
        user_prompt=f"1. 这个人是否有带安全帽 2. 这个人是否穿防护服(胸口带反光条) 3. 这个人是否穿全黑色鞋子(其他颜色都不行,带白色边或白色底也不行) 返回json, yes or no or unknown(如果无法判断)",
        # result_json_format=True,  # return any JSON
        
        # result_json_format with schema, When the vocabulary size of the result is smaller, the inference process will be more efficient.
        result_json_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ppe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "safety_helmet": {"type": "string"},
                        "reflective_clothes": {"type": "string"},
                        "black_shoes": {"type": "string"}, 
                    },
                    "required": ["safety_helmet", "reflective_clothes", "black_shoes"],
                },
            },
        },
    )

    # save the result to the local file
    my_node = my_node(node_name="my_node", save_path="/mnt/d/save_path")
    insight = CPipeInsight()

    streamer += [ce_det, vlm, my_node, insight]

    Node.launch()