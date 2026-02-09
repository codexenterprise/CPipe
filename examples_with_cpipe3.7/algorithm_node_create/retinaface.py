from cpipe.module.model.retinaface import Retinaface

if __name__ == "__main__":

    rf = Retinaface(
        "retinaface",
        "../../src/model_files/416x416-det_10g_batch.engine",
        3,
        (3, 416, 416),
        ["face"],
        max_batch_size=64,
        secondary_class_names=["person"],
    )