from cpipe.module.model.facerecognition import FaceRecognition

if __name__ == "__main__":

    fr = FaceRecognition(
        "adaface",
        "../../src/model_files/adaface_ir101_webface12m_batch64_GPU3070.engine",
        3,
        [3, 112, 112],
        max_batch_size=64,
        # face_quality_model_path="../../src/model_files/face_quality_batch64_GPU3070.engine",
        secondary_class_names=["person"],
        faces_library=fl,
        matching_score_thresh=0.1
    )
