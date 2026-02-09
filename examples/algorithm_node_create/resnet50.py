from cpipe.module.model.resnet50 import MMResnet50

if __name__ == "__main__":

    dianbiao = MMResnet50(
        model_path="./models/dianbiao_192_4.30_batch3.om",
        input_size=(3, 192, 192),
        max_batch_size=3,
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="./models/center_weight_origin.npy",
        num_joints=6,
        max_batch_size=3,
        secondary_class_names=['电流表', '电压表'],
    )
