from cpipe.module.model.retinaface import Retinaface

if __name__ == "__main__":

    rf = Retinaface(
        model_path="../../src/model_files/416x416-det_10g_batch.engine",
        input_size=(3, 416, 416),
        secondary_class_names=["person"],
    )