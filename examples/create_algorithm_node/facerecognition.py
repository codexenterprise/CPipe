from cpipe.module.model.facerecognition import FaceRecognition

if __name__ == "__main__":

    fr = FaceRecognition(
        model_path="../../src/model_files/adaface_ir101_webface12m_batch64_GPU3070.engine",
        max_batch_size=64,
        secondary_class_names=["person"],
    )
