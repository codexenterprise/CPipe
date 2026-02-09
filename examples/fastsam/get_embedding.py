import pickle

import cv2

from cpipe.module.model.dino import DinoEmbedding

if __name__ == "__main__":
    dino = DinoEmbedding(
        node_name="dino",
        model_path="/mnt/d/embed_model.onnx",
        input_size=(3, 224, 224),
        max_batch_size=8,
        warmup=True,
        device="cuda:0",
        need_embedding=True
    )

    dino.load_model()

    frame = cv2.imread("./img.png")

    e = dino([frame])
    # to pick the first image
    e = e[0]

    # save the embedding
    with open("embedding.pkl", "wb") as f:
        pickle.dump(e, f)

    print(e)

    # load the embedding
    with open("embedding.pkl", "rb") as f:
        e = pickle.load(f)

    print(e)