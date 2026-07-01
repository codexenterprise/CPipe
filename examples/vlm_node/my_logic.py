import multiprocessing
import os
import threading
import time
import cv2
from cpipe.module.cdata import LLMInfo
from cpipe.module.logic import Logic
from cpipe.module.node import Node


class my_node(Logic):
    def __init__(self, save_path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _start(self):

        while True:
            new_cdata = self.get_frames()

            det_stream_names, det_bboxes = new_cdata.get_bboxes()

            for idx, one_image_boxes in enumerate(det_bboxes):
                frame_streamer_name = det_stream_names[idx]
                img, _ = self.get_streamer_frame(cimage=new_cdata.images[frame_streamer_name])
                one_image_boxes = det_bboxes[idx]
                for one_box in one_image_boxes:
                    if one_box.llm_info is None:
                        self.logger.error("LLMInfo is None")
                        continue
                    self.logger.info(one_box.llm_info.to_json())
                    if self.save_path is not None and one_box.llm_info.response is not None:
                        cut_img = img[int(one_box.box_coord[1]):int(one_box.box_coord[3]), int(one_box.box_coord[0]):int(one_box.box_coord[2])].copy()
                        safety_helmet = one_box.llm_info.response.get("safety_helmet", "")
                        reflective_clothes = one_box.llm_info.response.get("reflective_clothes", "")
                        black_shoes = one_box.llm_info.response.get("black_shoes", "")
                        file_name = f"{time.time()}_{safety_helmet}_{reflective_clothes}_{black_shoes}.jpg"
                        # save the image to the local file
                        cv2.imwrite(os.path.join(self.save_path, file_name), cut_img)

            self.queue.put_cdata(new_cdata)




class my_node_with_to_vlm(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FORKLIFT_PROMPT = """ 你是一位经验丰富的工厂安全监控复核员。请仔细分析这张监控截图，判断叉车附近是否有人员靠近。
【场景定义】
● 叉车：带有货叉的工业搬运车辆，正在移动或静止作业
● 司机：坐在叉车驾驶座上、正在操作叉车的人员，属于正常工作状态，不是行人
● 行人：叉车外部步行的人员，或站在叉车附近区域的人员

【观察重点】（按优先级排序）
1. 叉车位置：画面中是否有叉车？叉车是否在移动或作业状态？
2. 人员位置：人员是否在叉车驾驶座上（司机）？还是在叉车附近地面上/旁边？
3. 距离判断：叉车外部的人员与叉车之间的距离是否过近（进入叉车作业危险区域）？

【严格排除以下情形】 | 情形 | 说明 | |:---|:---| | 叉车司机 | 坐在驾驶座上操作叉车的人员，属于正常工作 | | 另一台叉车的司机 | 旁边叉车上的操作人员，不是行人 | | 远处背景人物 | 与叉车有明显距离，不在作业危险区域内 | | 固定岗位人员 | 站在安全线外或固定工位上，未靠近叉车 | | 叉车维修人员 | 车辆静止且明显在检修，非运行状态 |
【输出格式】 必须且只输出文本： 有行人靠近运行中的叉车则只输出一行小写 yes，否则只输出一行小写 no。 """

    def _start(self):

        while True:
            new_cdata = self.get_frames()

            llm_info = LLMInfo(
                user_prompt=self.FORKLIFT_PROMPT,
                file_path="/mnt/d/__dataset__/cche/106.png",
            )

            new_cdata.llm_infos ={
                "my_node_with_to_vlm": [llm_info]
            }

            self.queue.put_cdata(new_cdata)


class my_node_report(Logic):
    def __init__(self, save_path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _start(self):

        while True:
            new_cdata = self.get_frames()

            for llm_info in new_cdata.llm_infos.get("my_node_with_to_vlm", []):
                if llm_info.response is not None:
                    self.logger.info(llm_info.response)
                    # to send the response to the other system
                else:
                    self.logger.error("LLMInfo response is None")

            self.queue.put_cdata(new_cdata)



class my_node_with_to_vlm_event_mode(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_queue = multiprocessing.Queue(maxsize=1024)
        self.FORKLIFT_PROMPT = """ 你是一位经验丰富的工厂安全监控复核员。请仔细分析这张监控截图，判断叉车附近是否有人员靠近。
【场景定义】
● 叉车：带有货叉的工业搬运车辆，正在移动或静止作业
● 司机：坐在叉车驾驶座上、正在操作叉车的人员，属于正常工作状态，不是行人
● 行人：叉车外部步行的人员，或站在叉车附近区域的人员

【观察重点】（按优先级排序）
1. 叉车位置：画面中是否有叉车？叉车是否在移动或作业状态？
2. 人员位置：人员是否在叉车驾驶座上（司机）？还是在叉车附近地面上/旁边？
3. 距离判断：叉车外部的人员与叉车之间的距离是否过近（进入叉车作业危险区域）？

【严格排除以下情形】 | 情形 | 说明 | |:---|:---| | 叉车司机 | 坐在驾驶座上操作叉车的人员，属于正常工作 | | 另一台叉车的司机 | 旁边叉车上的操作人员，不是行人 | | 远处背景人物 | 与叉车有明显距离，不在作业危险区域内 | | 固定岗位人员 | 站在安全线外或固定工位上，未靠近叉车 | | 叉车维修人员 | 车辆静止且明显在检修，非运行状态 |
【输出格式】 必须且只输出文本： 有行人靠近运行中的叉车则只输出一行小写 yes，否则只输出一行小写 no。 """
    

    def _listen_event_queue(self):
        while True:
            event = self.send_queue.get()
            start_time = time.time()
            ret = Node.__allNodes__["qwen_vlm"].event_send("chat", event)
            print(ret)
            end_time = time.time()
            print(f"Time cost: {end_time - start_time} seconds")

    def _start(self):
        # create a thread to listen to the event queue
        threading.Thread(target=self._listen_event_queue, daemon=True).start()

        while True: 
            new_cdata = self.get_frames()

            # path mode
            llm_info = LLMInfo(
                user_prompt=self.FORKLIFT_PROMPT,
                file_path="/mnt/d/__dataset__/cche/106.png",
            )

            # np.ndarray mode
            # img = cv2.imread("/mnt/d/__dataset__/cche/106.png")
            # llm_info = LLMInfo(
            #     user_prompt=self.FORKLIFT_PROMPT,
            #     file_path=img,
            # )

            # put the llm_info to the event queue
            self.send_queue.put({"question1": llm_info})

            self.queue.put_cdata(new_cdata)
    
