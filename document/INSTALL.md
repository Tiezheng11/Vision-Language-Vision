# Installation
### Create Environments
```bash
conda create -n VLV python=3.10.16
conda activate VLV
cd VLV-AutoEncoder-Official/
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Download Pretrained Weights
```bash
cd VLV-AutoEncoder-Official/models
git clone https://huggingface.co/microsoft/Florence-2-large
mv Florence-2-large Florence2large
cd ../
cd pretrained_checkpoints/
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
git clone https://huggingface.co/Qwen/Qwen2.5-3B
cd ..
```