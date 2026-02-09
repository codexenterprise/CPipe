import pickle

import torch

from cpipe.module.model.dino import DinoEmbedding
from cpipe.module.model.fastsam import FastSAM
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":

    video_zhu = "/mnt/d/videos/dengpao/hei/1802121J_20240323080000-20240323081500_1.mp4"

    stream_zhu = VideoStreamer("zhushitu", video_zhu, 3, 1, once_mode=False)

    fsam = FastSAM(
        "fastsam",
        "./src/model_files/FastSAM-x.engine",
        3,
        (3, 576, 1024),
        max_batch_size=1,
        conf_thres=0.80,
        device="cuda:0"
    )

    skuid_list = ["电流表"]
    with open("./examples/sam/embedding.pkl", "rb") as f:
        embeddings = pickle.load(f)
    embeddings = embeddings[None, ...]

    dino = DinoEmbedding(
        "dino",
        "./src/model_files/dino_embedding.engine",
        3,
        inputSize=(3, 224, 224),
        class_names=skuid_list,
        max_batch_size=32,
        warmup=True,
        device="cuda:0",
        secondary_class_names=['?'],
        embeddings=embeddings,
        need_embedding=False,
        conf_thres=0.40
    )


    cpipeinsight = CPipeInsight(http_insight=True)

    stream_zhu += [fsam, dino, cpipeinsight]

    Node.launch(check_node=True)