import os
import cv2

from cpipe.module.cdata import CData
from cpipe.module.model.retinaface import RetinafaceTRT

person_images_path = "/mnt/d/dataset/NBJB"
person_images_files = os.listdir(person_images_path)

save_face_images_path = "./face_images"
if not os.path.exists(save_face_images_path):
    os.makedirs(save_face_images_path)

rf = RetinafaceTRT(
    "retinaface",
    "../../src/model_files/416x416-det_10g_batch.engine",
    3,
    (3, 416, 416),
    ["face"],
    max_batch_size=64,
    # secondary_class_names=["person"],
)

rf._loadModel()

for idx, one in enumerate(person_images_files):
    one_path = os.path.join(person_images_path, one)
    img = cv2.imread(one_path)
    frames = [img]
    new_cdata = CData(rf.nodeName)
    pred = rf.forward(frames)
    rf.to_cdata(pred, new_cdata, frames, ["1"])
    for one_box in new_cdata.bboxes["1"]:
        face_img = img[int(one_box.box_coord[1]):int(one_box.box_coord[3]), int(one_box.box_coord[0]):int(one_box.box_coord[2])]
        cv2.imwrite(os.path.join(save_face_images_path, one), face_img)
    print(f"{idx}/{len(person_images_files)} done")
