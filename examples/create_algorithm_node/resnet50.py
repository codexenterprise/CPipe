from cpipe.module.model.resnet50 import MMResnet50

if __name__ == "__main__":

    dianbiao = MMResnet50(
        model_path="./models/dianbiao_192_4.30_batch3.om",
        class_names=['1', "2", "3", "4", "5", "6"],
        secondary_class_names=['电流表', '电压表'],
    )
