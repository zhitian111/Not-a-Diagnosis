$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:TORCH_USE_CUDA_DSA="1"
python main.py `
    --epochs 100 `
    --method ResNet `
    --exp_name ResNet_100epochs `
    --batch_size 2