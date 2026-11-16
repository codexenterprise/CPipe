"""Decider 接入示例。模型转换见同目录 description.md、export_onnx.py、to_trt.py。

320 letterbox：3070 ORT INT8 ~2FPS，4090 TRT FP16 ~10FPS。
embed 始终用 ONNX CPU，不要填 .engine。
"""
import json

import cpipe as cp
from cpipe.module.insight import CPipeInsight
from cpipe.module.model.decider import Decider
from cpipe.module.model.yolo26 import YOLO26
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer


@cp.logic
def print_decider(frame: cp.Frame):
    for box in frame.boxes:
        info = box.llm_info
        if info is None or info.response is None:
            continue
        questions = info.user_prompt or {}
        for qid, ans in info.response.items():
            q = questions.get(qid, {})
            print(f"[{frame.streamer_name}] {qid}  问题={q.get('instructions', q)}  结果={json.dumps(ans, ensure_ascii=False)}")

DECIDER = "/home/zh/workspace/cpipe2.0/project/decider-onnx"

if __name__ == "__main__":
    stream = VideoStreamer(stream="rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101", process_frame_interval=5)

    # det = YOLO26(
    #     model_path="model_files/yolo26n.engine",
    #     valid_class_names=["person"],
    # )

    decider = Decider(  # 320: 3070 ORT ~2FPS, 4090 TRT ~10FPS
        vision_model_path=f"{DECIDER}/out-int8/vision.onnx",
        embed_model_path=f"{DECIDER}/out-int8/embed.onnx",
        decoder_model_path=f"{DECIDER}/out-int8/decoder_t320.onnx",
        ckpt_path=f"{DECIDER}/ckpt",
        input_size=[3, 320, 320],
        # 外层 key 选「对谁提问」，取一份：box_class_name > streamer名 > "default"
        # value 是一道题，或一组命名题 {qid: 题}（当前 decoder 最多 Q=2）
        # 题型:
        #   noul   是非，答案 noul=P(true)；criteria 可选 {"true": "...", "false": "..."}
        #   choice 单选，criteria 的 key 就是选项名
        #   score  有序打分，criteria 是从低到高的档位列表
        questions={
            # 一阶段：整帧。noul 只能问是否，不能问“几个人”
            "default": {
                "type": "noul",
                "instructions": "这个人有没有在玩手机?",
                "criteria": {"true": "玩手机", "false": "没玩手机"},
            },
            # 一阶段多题（最多 Q 道）:
            # "default": {
            #     # "has_person": {"type": "noul", "instructions": "画面中有人吗?"},
            #     "count": {
            #         "type": "choice",
            #         "instructions": "画面中大约几个人?",
            #         "criteria": {"0": "没有", "1": "一个人", "2": "两个人", "3": "三个人", "4": "四个人", "5": "五个人", ">5": "超过五个人"},
            #     },
            # },
            # 二阶段：对 person 框提问（打开上面 YOLO，并设 secondary_class_names=["person"]）
            # "person": {
            #     "type": "choice",
            #     "instructions": "这个人在做什么?",
            #     "criteria": {"stand": "站着", "sit": "坐着", "walk": "走路", "other": "其他"},
            # },
        },
        # secondary_class_names=["person"], # 二阶段模式下可以接检测节点后面就行人员分类
        gpu_mem_limit_gb=6.0,
    )

    insight = CPipeInsight(http_insight=True)
    stream += [decider, print_decider(), insight]
    Node.launch(check_node=True, node_ready_timeout=180)
