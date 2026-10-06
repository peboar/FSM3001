from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

import config_unet as cfg
from ct.ct_texture import ExtractCtTexture
from unet.unet import UNet
from unet.unet_data import SingleUnetDataset, compute_class_weights


if __name__ == "__main__":
    project_path = Path(__file__).resolve().parent
    data_path = project_path / "data" / (cfg.AGGREGATE_TYPE + "s")
    ct_path = project_path / "ct"

    phases = cfg.PHASES

    textures = {}

    for phase in phases:
        texture_path = ct_path / phase

        texture = ExtractCtTexture(
            folder_path=texture_path,
            image_size=cfg.IMAGE_SIZE,
            image_extension=cfg.IMAGE_EXTENSION,
        )

        textures[phase] = texture

    phase_colors = {
        phase: texture.extract_gray_mean()
        for phase, texture in textures.items()
    }

    material_inclusions = cfg.MATERIAL_INCLUSIONS

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")

    packing_directories = [
        path
        for path in sorted(data_path.iterdir())
        if path.is_dir()
    ]

    generator = torch.Generator().manual_seed(
        cfg.RANDOM_SEED
    )

    number_of_packings = len(packing_directories)

    number_of_training_packings = int(
        cfg.TRAIN_RATIO * number_of_packings
    )

    number_of_validation_packings = int(
        cfg.VALIDATION_RATIO * number_of_packings
    )

    number_of_testing_packings = (
        number_of_packings
        - number_of_training_packings
        - number_of_validation_packings
    )

    (
        training_packings,
        validation_packings,
        testing_packings,
    ) = random_split(
        packing_directories,
        [
            number_of_training_packings,
            number_of_validation_packings,
            number_of_testing_packings,
        ],
        generator=generator,
    )

    training_datasets = [
        SingleUnetDataset(
            packing_path=packing,
            image_size=cfg.IMAGE_SIZE,
            image_extension=cfg.IMAGE_EXTENSION,
            phase_colors=phase_colors,
            material_inclusions=material_inclusions,
            edge_factor=cfg.EDGE_FACTOR,
            seed=None,
        )
        for packing in training_packings
    ]

    validation_datasets = [
        SingleUnetDataset(
            packing_path=packing,
            image_size=cfg.IMAGE_SIZE,
            image_extension=cfg.IMAGE_EXTENSION,
            phase_colors=phase_colors,
            material_inclusions=material_inclusions,
            edge_factor=cfg.EDGE_FACTOR,
            seed=i + cfg.RANDOM_SEED,
        )
        for i, packing in enumerate(validation_packings)
    ]

    testing_datasets = [
        SingleUnetDataset(
            packing_path=packing,
            image_size=cfg.IMAGE_SIZE,
            image_extension=cfg.IMAGE_EXTENSION,
            phase_colors=phase_colors,
            material_inclusions=material_inclusions,
            edge_factor=cfg.EDGE_FACTOR,
            seed=i + cfg.RANDOM_SEED,
        )
        for i, packing in enumerate(testing_packings)
    ]

    training_dataset = ConcatDataset(
        training_datasets
    )

    validation_dataset = ConcatDataset(
        validation_datasets
    )

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE_TRAINING,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE_VALIDATION,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
    )

    model = UNet(
        cfg.IN_CHANNELS,
        cfg.NUM_CLASSES,
    )

    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
    )

    class_weights = compute_class_weights(
        training_packings,
        cfg.IMAGE_EXTENSION,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    checkpoint_path = (
        data_path / "unet_phase_randomized.pth"
    )

    best_validation_loss = float("inf")

    for epoch in tqdm(range(cfg.NUM_EPOCHS)):
        model.train()

        running_loss_training = 0.0

        for image, mask in training_dataloader:
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(
                output,
                mask,
            )

            loss.backward()
            optimizer.step()

            running_loss_training += loss.item()

        loss_training = (
            running_loss_training
            / len(training_dataloader)
        )

        model.eval()

        running_loss_validation = 0.0

        with torch.no_grad():
            for image, mask in validation_dataloader:
                image = image.to(device)
                mask = mask.to(device)

                output = model(image)

                loss = criterion(
                    output,
                    mask,
                )

                running_loss_validation += loss.item()

        loss_validation = (
            running_loss_validation
            / len(validation_dataloader)
        )

        tqdm.write(
            f"epoch {epoch}: "
            f"train {loss_training:.4f}, "
            f"val {loss_validation:.4f}"
        )

        if loss_validation < best_validation_loss:
            best_validation_loss = loss_validation

            torch.save(
                model.state_dict(),
                str(checkpoint_path),
            )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.eval()

    true_positive = np.zeros(
        cfg.NUM_CLASSES,
        dtype=np.int64,
    )

    predicted_pixels = np.zeros(
        cfg.NUM_CLASSES,
        dtype=np.int64,
    )

    ground_truth_pixels = np.zeros(
        cfg.NUM_CLASSES,
        dtype=np.int64,
    )

    for i, packing in enumerate(testing_packings):
        packing = Path(packing)

        dataset = testing_datasets[i]

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        inference_path = packing / "inference"
        inference_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        if cfg.CLEAR_INFERENCE:
            for file in inference_path.glob("*"):
                if file.is_file():
                    file.unlink()

        print(
            f"Generating inference plots for "
            f"{packing.name} "
            f"({i + 1}/{number_of_testing_packings})"
        )

        with torch.no_grad():
            for index, (image, mask) in enumerate(loader):
                image = image.to(device)

                output = model(image)

                prediction = (
                    torch.argmax(
                        output,
                        dim=1,
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                ground_truth = (
                    mask
                    .squeeze(0)
                    .numpy()
                )

                for class_index in range(
                    cfg.NUM_CLASSES
                ):
                    true_class = (
                        ground_truth == class_index
                    )

                    predicted_class = (
                        prediction == class_index
                    )

                    true_positive[class_index] += np.sum(
                        true_class & predicted_class
                    )

                    predicted_pixels[class_index] += np.sum(
                        predicted_class
                    )

                    ground_truth_pixels[class_index] += np.sum(
                        true_class
                    )

                prediction_mask = np.zeros_like(
                    prediction,
                    dtype=np.uint8,
                )

                prediction_mask[
                    prediction == 1
                ] = 128

                prediction_mask[
                    prediction == 2
                ] = 255

                original_name = (
                    dataset.slices[index].name
                )

                inference_name = (
                    original_name.replace(
                        "slice",
                        "inference",
                        1,
                    )
                )

                Image.fromarray(
                    prediction_mask
                ).save(
                    inference_path / inference_name
                )

    denominator = (
        predicted_pixels
        + ground_truth_pixels
    )

    dice = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(
            true_positive,
            dtype=np.float64,
        ),
        where=denominator != 0,
    )

    overall_dice = np.mean(dice)

    print("\nTest results")
    print("============")
    print(f"Overall Dice:    {overall_dice:.4f}")
    print(f"Void Dice:       {dice[0]:.4f}")
    print(f"Aggregate Dice:  {dice[1]:.4f}")
    print(f"Boundary Dice:   {dice[2]:.4f}")