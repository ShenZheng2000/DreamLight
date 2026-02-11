# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# pip install numpy==1.26.4
# for `ckpt/SD15/vqmodel/config.json`, check: https://github.com/yongliu20/DreamLight/issues/7

# python SD15/test.py

CUDA_VISIBLE_DEVICES=0 python SD15/test.py --relight_type noon_sunlight_1 --image_width 512 --image_height 512 &
CUDA_VISIBLE_DEVICES=1 python SD15/test.py --relight_type golden_sunlight_1 --image_width 512 --image_height 512 &
CUDA_VISIBLE_DEVICES=2 python SD15/test.py --relight_type foggy_1 --image_width 512 --image_height 512 &
CUDA_VISIBLE_DEVICES=3 python SD15/test.py --relight_type moonlight_1 --image_width 512 --image_height 512 &
wait