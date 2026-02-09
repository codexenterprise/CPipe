from cpipe.module.insight import CPipeInsight
from cpipe.module.model.sam3 import SAM3
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer


if __name__ == "__main__":

    stream = VideoStreamer(stream="rtmp://192.168.10.7:1935/live/7777", process_frame_interval=1)
    sam3 = SAM3(
            # prompt="shoe",
            boxes=[[1007, 550, 1132, 718]],
            box_labels=[1],
            decoder_model_path="/mnt/d/models/sam3/decoder-fp16.engine", 
            vision_encoder_model_path="/mnt/d/models/sam3/vision-encoder-fp16.engine", 
            text_encoder_model_path="/mnt/d/models/sam3/text-encoder-fp16.engine", 
            geometry_encoder_model_path="/mnt/d/models/sam3/geometry-encoder-fp16.engine",
            tokenizer_file_path="/mnt/d/models/sam3/tokenizer.json",
            conf_thres=0.4,
        )


    cpipeinsight = CPipeInsight(http_insight=True, chinese_font_size=30)
    stream += [sam3, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, check_interval=5)



