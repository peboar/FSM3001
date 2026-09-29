## Windows Quick Start

### 1. Add Blender to PATH
Add this path to your Windows Environment Variables (**Path**):
`C:\Program Files\Blender Foundation\Blender 5.1`
*(Restart your terminal afterward).*

### 2. Environment Setup
Open **Anaconda Prompt** and run:
```bash
# Create and register environment
conda create --prefix C:\Users\boekhout\ct python=3.11 -y
conda config --append envs_dirs C:\Users\boekhout
conda activate ct

# Install PyTorch (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install dependencies
conda install -c conda-forge shapely trimesh albumentations opencv scipy pillow matplotlib noise tqdm -y
```

### 3. Run Scripts
```bash
conda activate ct

# Run 3D Packing (Headless Blender)
blender -b -P main_packing.py

# Run U-Net Training
python main_unet.py
```
