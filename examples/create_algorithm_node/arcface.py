from cpipe.module.model.arcface import Arcface

if __name__ == "__main__":
    fr = Arcface(
        model_path="../../src/model_files/adaface.engine",
        max_batch_size=64,
        secondary_class_names=["person"],
    )

