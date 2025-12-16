$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:TORCH_USE_CUDA_DSA="1"
python main.py `
    --mode eval `
    --method ResNet `
    --ckpt_path .\train_results\20251215-233951\models\ResNet\checkpoints\best.pth