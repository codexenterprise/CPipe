import os
import pickle

import cv2

from cpipe.module.cdata import CData
from cpipe.module.model.adaface import Adaface

face_images_path = "./face_images"
face_images_files = os.listdir(face_images_path)

save_embedding_images_path = "./face_embeddings"


ada = Adaface(
    "adaface",
    "../../src/model_files/adaface_ir101_webface12m_batch64_GPU3070.engine",
    3,
    [3, 112, 112],
    max_batch_size=8,
    face_quality_model_path="../../src/model_files/face_quality_batch64_GPU3070.engine",
)

ada._loadModel()

for idx, one in enumerate(face_images_files):
    if one.endswith(".pkl"):
        continue
    one_path = os.path.join(face_images_path, one)
    img = cv2.imread(one_path)
    frames = [img]
    new_cdata = CData(ada.nodeName)
    pred = ada.forward(frames, box_kps=None)
    ada.to_cdata(pred, new_cdata, frames, ["1"])
    fe = new_cdata.bboxes['1'][0].person.face_embedding
    # pickle.dump(fe, open(os.path.join(save_embedding_images_path, one.replace('.jpg', '.pkl')), "wb"))
    with open(os.path.join(save_embedding_images_path, one.replace('.jpg', '.pkl')), "wb") as f:
        pickle.dump(fe, f)
    print(f"{idx}/{len(face_images_files)} done")
