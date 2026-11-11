from transformers.trainer_utils import TrainOutput


pivotedQR = TrainOutput(
    global_step=10000,
    training_loss=1.190963525390625,
    metrics={
        "train_runtime": 6407.1181,
        "train_samples_per_second": 12.486,
        "train_steps_per_second": 1.561,
        "total_flos": 2.0820092201385984e16,
        "train_loss": 1.190963525390625,
        "epoch": 12.5,
    },
)


naive = TrainOutput(
    global_step=10000,
    training_loss=1.189498828125,
    metrics={
        "train_runtime": 6393.7748,
        "train_samples_per_second": 12.512,
        "train_steps_per_second": 1.564,
        "total_flos": 2.0820092201385984e16,
        "train_loss": 1.189498828125,
        "epoch": 12.5,
    },
)


svd = TrainOutput(
    global_step=10000,
    training_loss=1.250710968017578,
    metrics={
        "train_runtime": 6431.9995,
        "train_samples_per_second": 12.438,
        "train_steps_per_second": 1.555,
        "total_flos": 2.0820092201385984e16,
        "train_loss": 1.250710968017578,
        "epoch": 12.5,
    },
)


lora = TrainOutput(
    global_step=10000,
    training_loss=1.2239700073242188,
    metrics={
        "train_runtime": 6331.801,
        "train_samples_per_second": 12.635,
        "train_steps_per_second": 1.579,
        "total_flos": 2.082637886644224e16,
        "train_loss": 1.2239700073242188,
        "epoch": 12.5,
    },
)
