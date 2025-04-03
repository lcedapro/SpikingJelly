echo "CUDA GPU 训练示例脚本"
echo "需要安装torch spikingjelly time os argparse numpy等库"
echo "-device, -amp, -cupy: 都不能动"
echo "-data_dir: 数据集路径，需要指定。该路径下有0、1、2等子文件夹，每个子文件夹下有若干张图片"
echo "-out_dir: 输出路径，需要指定。用于存储训练日志和模型"
echo "-channels: 通道数，为2，跟模型有关，不要动"
echo "-opt, -lr_scheduler, -b, -T_max, -epochs, -lr: 训练参数，根据需要调整，详见davispku_duration_2000_static.py"


python davispku_duration_2000_static.py -device cuda:0 -amp -cupy -data_dir './temporary_datasets/duration_2000_0306' -out_dir './logs_t1e4_simple' -channels 2 -opt SGD -lr_scheduler CosALR -b 64 -T_max 48 -epochs 48 -lr 0.4
