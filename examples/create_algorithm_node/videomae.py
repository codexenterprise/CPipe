from cpipe.module.model.videomae import VideoMAE

if __name__ == "__main__":
    detect = VideoMAE(
                    model_path = "project/wyt/model_files/wyt2.5_320.engine.cpipe",
                    input_size=(3, 320, 320),
                    max_batch_size=3,
                    )
