from cpipe.module.model.movenet import MoveNet

if __name__ == "__main__":

    dianbiao = MoveNet(
        node_name="dianbiao",
        model_path="./models/dianbiao_192_4.30_batch3.om",
        queue_size=3,
        input_size=(3, 192, 192),
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="./models/center_weight_origin.npy",
        num_joints=6
    )
