from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

import config_unet as cfg
from ct import ct_texture
from ct.ct_texture import ExtractCtTexture
from unet.unet import UNet
from unet.unet_data import SingleUnetDataset, compute_class_weights


project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / cfg.DATA_TYPE
ct_path = project_path / "ct"

# Obtain the colors for all the phases. Required for the augmentation step
phases = cfg.PHASES

textures = {}
for phase in phases:
    texture_path = ct_path / phase

    texture = ExtractCtTexture(
        folder_path=texture_path,
        image_size=cfg.IMAGE_SIZE,
        image_extension=cfg.IMAGE_EXTENSION
    )
    textures[phase] = texture

phase_colors = {
    phase: texture.extract_mean_gray()
    for phase, texture in textures.items()
}

ct_textures = {
    phase: texture.extract_ct_texture()
    for phase, texture in textures.items()
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

packing_directories = [
    path
    for path in sorted(data_path.iterdir())
    if path.is_dir()
]

generator = torch.Generator().manual_seed(cfg.RANDOM_SEED)

# Split into training, validation and testing sets
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

training_packings, validation_packings, testing_packings = random_split(
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
        noise_type=cfg.NOISE_TYPE,
        phase_colors=phase_colors,
        ct_textures=ct_textures,
    )
    for packing in training_packings
]

validation_datasets = [
    SingleUnetDataset(
        packing_path=packing,
        image_size=cfg.IMAGE_SIZE,
        noise_type=cfg.NOISE_TYPE,
        phase_colors=phase_colors,
        ct_texture=ct_textures,
    )
    for packing in validation_packings
]

testing_datasets = [
    SingleUnetDataset(
        packing_path=packing,
        image_size=cfg.IMAGE_SIZE,
        noise_type=cfg.NOISE_TYPE,
        phase_colors=phase_colors,
        ct_texture=ct_textures,
    )
    for packing in testing_packings
]

# Merge individual datasets into continuous datasets
training_dataset = ConcatDataset(training_datasets)
validation_dataset = ConcatDataset(validation_datasets)
testing_dataset = ConcatDataset(testing_datasets)

training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
)

validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
)

testing_dataloader = DataLoader(
    dataset=testing_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
)

model = UNet(
    cfg.IN_CHANNELS,
    cfg.NUM_CLASSES,
)
# Ensure the model and input reside on the same device
model.to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=cfg.LEARNING_RATE,
)
# Weights used for the loss function
class_weights = compute_class_weights(training_packings).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

checkpoint_path = data_path / f"unet_noise_{cfg.NOISE_TYPE}.pth"
best_validation_loss = float("inf")

# Start training
for epoch in tqdm(range(cfg.NUM_EPOCHS)):
    model.train()

    running_loss_training = 0.0

    for image, mask in training_dataloader:
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, mask)

        loss.backward()
        optimizer.step()

        running_loss_training += loss.item()

    loss_training = (
        running_loss_training / len(training_dataloader)
    )

    model.eval()

    running_loss_validation = 0.0

    with torch.no_grad():
        for image, mask in validation_dataloader:
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)

            loss = criterion(output, mask)

            running_loss_validation += loss.item()

    loss_validation = (
        running_loss_validation / len(validation_dataloader)
    )

    tqdm.write(
        f"epoch {epoch}: "
        f"train {loss_training:.4f}, "
        f"val {loss_validation:.4f}"
    )

    if loss_validation < best_validation_loss:
        best_validation_loss = loss_validation
        torch.save(model.state_dict(), str(checkpoint_path))

