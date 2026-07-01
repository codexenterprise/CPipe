from cpipe.module.model.adaface import Adaface

if __name__ == "__main__":
    fr = Adaface(
        model_path="../../src/model_files/adaface.engine",
        max_batch_size=64,
        secondary_class_names=["person"],
    )

