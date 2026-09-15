import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # --- ENCODER ---
        # Level 1
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 2
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 3
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 4
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK ---
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # --- DECODER (Indexed by target level) ---
        # Level 4
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Level 3
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Level 2
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Level 1
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # --- ENCODER PATH ---
        # Level 1
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        # Level 2
        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        # Level 3
        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        # Level 4
        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        # --- BOTTLENECK ---
        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # --- DECODER PATH ---
        # Level 4 (Processes Bottleneck output + Encoder Level 4 skip)
        xu4 = self.upconv4(xe52)
        xu42 = torch.cat([xu4, xe42], dim=1)
        xd41 = relu(self.d41(xu42))
        xd42 = relu(self.d42(xd41))

        # Level 3 (Processes Level 4 output + Encoder Level 3 skip)
        xu3 = self.upconv3(xd42)
        xu32 = torch.cat([xu3, xe32], dim=1)
        xd31 = relu(self.d31(xu32))
        xd32 = relu(self.d32(xd31))

        # Level 2 (Processes Level 3 output + Encoder Level 2 skip)
        xu2 = self.upconv2(xd32)
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        # Level 1 (Processes Level 2 output + Encoder Level 1 skip)
        xu1 = self.upconv1(xd22)
        xu12 = torch.cat([xu1, xe12], dim=1)
        xd11 = relu(self.d11(xu12))
        xd12 = relu(self.d12(xd11))

        # Output Layer
        output = self.outconv(xd12)

        return output


if __name__ == "__main__":
    # 1. Initialize the model for 2 target classes (e.g., Background vs. Object)
    model = Unet(n_class=2)

    # 2. Create a dummy image tensor: [Batch size = 1, Channels = 1, Height = 256, Width = 256]
    # Note: U-Net requires image sizes divisible by 16 (like 256, 512, etc.) due to the 4 max-pools
    dummy_input = torch.randn(1, 1, 256, 256)

    # 3. Pass it through the model
    output = model(dummy_input)

    # 4. Check the results
    print("🎉 Success! The network executed perfectly without crashing.")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: [1, 2, 256, 256])")
