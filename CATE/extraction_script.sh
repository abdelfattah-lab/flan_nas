
####### CATE Extraction #######

# !!!! ALL SS !!!!
python inference/inference.py --pretrained_path model/all_ss_model_best.pth.tar --train_data data/all_ss/train_data.pt --valid_data data/all_ss/test_data.pt --dataset all_ss --search_space all_ss --n_vocab 20 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/all_ss_model_best.pth.tar --train_data data/all_ss/train_data.pt --valid_data data/all_ss/test_data.pt --dataset all_ss --search_space all_ss --n_vocab 20 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

# NASBench101
python inference/inference.py --pretrained_path model/nasbench101_model_best.pth.tar --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101 --search_space nasbench101 --n_vocab 5 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/nasbench101_model_best.pth.tar --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101 --search_space nasbench101 --n_vocab 5 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

# NASBench201
python inference/inference.py --pretrained_path model/nasbench201_model_best.pth.tar --train_data data/nasbench201/train_data.pt --valid_data data/nasbench201/test_data.pt --dataset nasbench201 --search_space nasbench201 --n_vocab 7 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/nasbench201_model_best.pth.tar --train_data data/nasbench201/train_data.pt --valid_data data/nasbench201/test_data.pt --dataset nasbench201 --search_space nasbench201 --n_vocab 7 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

# NASBench301
python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301 --search_space nasbench301 --n_vocab 11 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301 --search_space nasbench301 --n_vocab 11 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

# NASBench301 NR
python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nb301a/train_data.pt --valid_data data/nb301a/test_data.pt --dataset nb301a --search_space nb301a --n_vocab 11 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nb301a/train_data.pt --valid_data data/nb301a/test_data.pt --dataset nb301a --search_space nb301a --n_vocab 11 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nb301b/train_data.pt --valid_data data/nb301b/test_data.pt --dataset nb301b --search_space nb301b --n_vocab 11 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nb301b/train_data.pt --valid_data data/nb301b/test_data.pt --dataset nb301b --search_space nb301b --n_vocab 11 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

# NDS
python inference/inference.py --pretrained_path model/nds_Amoeba_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_Amoeba_normal_train_data.pt --valid_data data/nds/nds_Amoeba_normal_test_data.pt --dataset nds --search_space Amoeba --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_Amoeba_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_Amoeba_normal_train_data.pt --valid_data data/nds/nds_Amoeba_normal_test_data.pt --dataset nds --search_space Amoeba --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_Amoeba_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_Amoeba_reduce_train_data.pt --valid_data data/nds/nds_Amoeba_reduce_test_data.pt --dataset nds --search_space Amoeba --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_Amoeba_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_Amoeba_reduce_train_data.pt --valid_data data/nds/nds_Amoeba_reduce_test_data.pt --dataset nds --search_space Amoeba --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_PNAS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_PNAS_fix-w-d_normal_test_data.pt --dataset nds --search_space PNAS_fix-w-d --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_PNAS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_PNAS_fix-w-d_normal_test_data.pt --dataset nds --search_space PNAS_fix-w-d --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_PNAS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_PNAS_fix-w-d_reduce_test_data.pt --dataset nds --search_space PNAS_fix-w-d --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_PNAS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_PNAS_fix-w-d_reduce_test_data.pt --dataset nds --search_space PNAS_fix-w-d --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_ENAS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_ENAS_fix-w-d_normal_test_data.pt --dataset nds --search_space ENAS_fix-w-d --type normal --n_vocab 8 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_ENAS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_ENAS_fix-w-d_normal_test_data.pt --dataset nds --search_space ENAS_fix-w-d --type normal --n_vocab 8 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_ENAS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_ENAS_fix-w-d_reduce_test_data.pt --dataset nds --search_space ENAS_fix-w-d --type reduce --n_vocab 8 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_ENAS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_ENAS_fix-w-d_reduce_test_data.pt --dataset nds --search_space ENAS_fix-w-d --type reduce --n_vocab 8 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_NASNet_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_NASNet_normal_train_data.pt --valid_data data/nds/nds_NASNet_normal_test_data.pt --dataset nds --search_space NASNet --type normal --n_vocab 16 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_NASNet_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_NASNet_normal_train_data.pt --valid_data data/nds/nds_NASNet_normal_test_data.pt --dataset nds --search_space NASNet --type normal --n_vocab 16 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_NASNet_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_NASNet_reduce_train_data.pt --valid_data data/nds/nds_NASNet_reduce_test_data.pt --dataset nds --search_space NASNet --type reduce --n_vocab 16 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_NASNet_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_NASNet_reduce_train_data.pt --valid_data data/nds/nds_NASNet_reduce_test_data.pt --dataset nds --search_space NASNet --type reduce --n_vocab 16 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_DARTS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_normal_train_data.pt --valid_data data/nds/nds_DARTS_normal_test_data.pt --dataset nds --search_space DARTS --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_DARTS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_normal_train_data.pt --valid_data data/nds/nds_DARTS_normal_test_data.pt --dataset nds --search_space DARTS --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_DARTS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_reduce_train_data.pt --valid_data data/nds/nds_DARTS_reduce_test_data.pt --dataset nds --search_space DARTS --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_DARTS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_reduce_train_data.pt --valid_data data/nds/nds_DARTS_reduce_test_data.pt --dataset nds --search_space DARTS --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_ENAS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_normal_train_data.pt --valid_data data/nds/nds_ENAS_normal_test_data.pt --dataset nds --search_space ENAS --type normal --n_vocab 8 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_ENAS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_normal_train_data.pt --valid_data data/nds/nds_ENAS_normal_test_data.pt --dataset nds --search_space ENAS --type normal --n_vocab 8 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_ENAS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_reduce_train_data.pt --valid_data data/nds/nds_ENAS_reduce_test_data.pt --dataset nds --search_space ENAS --type reduce --n_vocab 8 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_ENAS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_reduce_train_data.pt --valid_data data/nds/nds_ENAS_reduce_test_data.pt --dataset nds --search_space ENAS --type reduce --n_vocab 8 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_PNAS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_normal_train_data.pt --valid_data data/nds/nds_PNAS_normal_test_data.pt --dataset nds --search_space PNAS --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_PNAS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_normal_train_data.pt --valid_data data/nds/nds_PNAS_normal_test_data.pt --dataset nds --search_space PNAS --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_PNAS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_reduce_train_data.pt --valid_data data/nds/nds_PNAS_reduce_test_data.pt --dataset nds --search_space PNAS --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_PNAS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_reduce_train_data.pt --valid_data data/nds/nds_PNAS_reduce_test_data.pt --dataset nds --search_space PNAS --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_DARTS_lr-wd_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_lr-wd_normal_train_data.pt --valid_data data/nds/nds_DARTS_lr-wd_normal_test_data.pt --dataset nds --search_space DARTS_lr-wd --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_DARTS_lr-wd_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_lr-wd_normal_train_data.pt --valid_data data/nds/nds_DARTS_lr-wd_normal_test_data.pt --dataset nds --search_space DARTS_lr-wd --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_DARTS_lr-wd_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_lr-wd_reduce_train_data.pt --valid_data data/nds/nds_DARTS_lr-wd_reduce_test_data.pt --dataset nds --search_space DARTS_lr-wd --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_DARTS_lr-wd_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_lr-wd_reduce_train_data.pt --valid_data data/nds/nds_DARTS_lr-wd_reduce_test_data.pt --dataset nds --search_space DARTS_lr-wd --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt


python inference/inference.py --pretrained_path model/nds_DARTS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_DARTS_fix-w-d_normal_test_data.pt --dataset nds --search_space DARTS_fix-w-d --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_DARTS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_DARTS_fix-w-d_normal_test_data.pt --dataset nds --search_space DARTS_fix-w-d --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

python inference/inference.py --pretrained_path model/nds_DARTS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_DARTS_fix-w-d_reduce_test_data.pt --dataset nds --search_space DARTS_fix-w-d --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
echo "python inference/inference.py --pretrained_path model/nds_DARTS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_DARTS_fix-w-d_reduce_test_data.pt --dataset nds --search_space DARTS_fix-w-d --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16" >> commands_log.txt

# TransNASBench101 Micro
python inference/inference.py --pretrained_path model/transnasbench101_autoencoder_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/autoencoder_train_data.pt --valid_data data/transnasbench101/autoencoder_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task autoencoder --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_autoencoder_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/autoencoder_train_data.pt --valid_data data/transnasbench101/autoencoder_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task autoencoder --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt


python inference/inference.py --pretrained_path model/transnasbench101_class_object_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/class_object_train_data.pt --valid_data data/transnasbench101/class_object_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task class_object --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_class_object_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/class_object_train_data.pt --valid_data data/transnasbench101/class_object_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task class_object --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt


python inference/inference.py --pretrained_path model/transnasbench101_class_scene_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/class_scene_train_data.pt --valid_data data/transnasbench101/class_scene_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task class_scene --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_class_scene_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/class_scene_train_data.pt --valid_data data/transnasbench101/class_scene_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task class_scene --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt


python inference/inference.py --pretrained_path model/transnasbench101_jigsaw_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/jigsaw_train_data.pt --valid_data data/transnasbench101/jigsaw_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task jigsaw --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_jigsaw_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/jigsaw_train_data.pt --valid_data data/transnasbench101/jigsaw_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task jigsaw --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt


python inference/inference.py --pretrained_path model/transnasbench101_normal_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/normal_train_data.pt --valid_data data/transnasbench101/normal_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task normal --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_normal_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/normal_train_data.pt --valid_data data/transnasbench101/normal_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task normal --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt


python inference/inference.py --pretrained_path model/transnasbench101_room_layout_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/room_layout_train_data.pt --valid_data data/transnasbench101/room_layout_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task room_layout --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_room_layout_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/room_layout_train_data.pt --valid_data data/transnasbench101/room_layout_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task room_layout --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt


python inference/inference.py --pretrained_path model/transnasbench101_segmentsemantic_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/segmentsemantic_train_data.pt --valid_data data/transnasbench101/segmentsemantic_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task segmentsemantic --n_vocab 6 --graph_d_model 32 --pair_d_model 32
echo "python inference/inference.py --pretrained_path model/transnasbench101_segmentsemantic_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/segmentsemantic_train_data.pt --valid_data data/transnasbench101/segmentsemantic_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task segmentsemantic --n_vocab 6 --graph_d_model 32 --pair_d_model 32" >> commands_log.txt

