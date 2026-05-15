扩充加随机误差方法：
# 先装依赖
pip install opencv-python imageio-ffmpeg

# 扩充到50条
python augment_lerobot.py \
    --src D:\mujuco\demos_arc_lerobot \
    --dst D:\mujuco\demos_arc_aug \
    --target 50 \
    --seed 42
