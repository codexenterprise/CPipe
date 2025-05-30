from cpipe.module.model.arcface import Arcface

if __name__ == "__main__":
    fr = Arcface(
        "adaface",
        "../../src/model_files/adaface.engine",
        3,
        [3, 112, 112],
        max_batch_size=64,
        # face_quality_model_path="../../src/model_files/face_quality_batch64_GPU3070.engine",
        secondary_class_names=["person"],
    )

