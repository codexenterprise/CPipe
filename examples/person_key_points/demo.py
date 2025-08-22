from cpipe.module.model.movenet import MoveNet

if __name__ == "__main__":

    dianbiao = MoveNet(
        "dianbiao",
        "./model/movenet_thorax_192_6.80_batch3.om",
        3,
        (3, 192, 192),
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="./models/center_weight_origin.npy",
        num_joints=6,
        max_batch_size=3,
        secondary_class_names=['电流表', '电压表'],
    )
