import os
import pickle
import cv2

from cpipe.module.model.adaface import Adaface

if __name__ == "__main__":
    face_images_path = "./face_images"
    face_images_files = os.listdir(face_images_path)

    save_embedding_images_path = "./face_embeddings"
    if not os.path.exists(save_embedding_images_path):
        os.makedirs(save_embedding_images_path)


    ada = Adaface(
        node_name="adaface",
        model_path="src/model_files/adaface_ir101_webface12m.engine",
        input_size=(3, 112, 112),
        max_batch_size=8,
        face_quality_model_path="src/model_files/face_quality_batch.onnx.cpipe",
    )

    ada.load_model()

    for idx, one in enumerate(face_images_files):
        if one.endswith(".pkl"):
            continue
        one_path = os.path.join(face_images_path, one)
        img = cv2.imread(one_path)
        frames = [img]
        cdata = ada(frames, return_cdata_format=True, frames_stream_names=["1"], box_kps=None)
        fe = cdata.bboxes["1"][0].person.face_embedding
        # pickle.dump(fe, open(os.path.join(save_embedding_images_path, one.replace('.jpg', '.pkl')), "wb"))
        with open(os.path.join(save_embedding_images_path, one.replace('.jpg', '.pkl')), "wb") as f:
            pickle.dump(fe, f)
        print(f"{idx}/{len(face_images_files)} done")
