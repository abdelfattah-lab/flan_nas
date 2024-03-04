python preprocessing/data_generate.py --dataset all_ss --flag build_pair --k 2 --d 50000 --metric params # 49979274 999585.48 --> 999585.48 is too high for several data-sets, thus for all_ss, we make maxDist 50k? 
echo "python preprocessing/data_generate.py --dataset all_ss --flag build_pair --k 2 --d 50000 --metric params # 49979274 999585.48 --> 999585.48 is too high for several data-sets, thus for all_ss, we make maxDist 50k? " >> commands_log.txt

python preprocessing/data_generate.py --dataset nasbench101 --flag build_pair --k 2 --d 9999 --metric params # 2000000/49979274 (4% in paper) [We reproduce at 2%]
echo "python preprocessing/data_generate.py --dataset nasbench101 --flag build_pair --k 2 --d 9999 --metric params # 2000000/49979274 (4% in paper) [We reproduce at 2%]" >> commands_log.txt

python preprocessing/data_generate.py --dataset nasbench201 --flag build_pair --k 2 --d 9999 --metric params # 1.53M 0.0306M this was maxing at 1.53, so make the distance 0.0306M (2%)
echo "python preprocessing/data_generate.py --dataset nasbench201 --flag build_pair --k 2 --d 9999 --metric params # 1.53M 0.0306M this was maxing at 1.53, so make the distance 0.0306M (2%)" >> commands_log.txt

python preprocessing/data_generate.py --dataset nasbench301 --flag build_pair --k 2 --d 9999 --metric params # 5000000/265754112.0 (2% in paper) Params: 1643754.0 32875.08
echo "python preprocessing/data_generate.py --dataset nasbench301 --flag build_pair --k 2 --d 9999 --metric params # 5000000/265754112.0 (2% in paper) Params: 1643754.0 32875.08" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space Amoeba  --type normal --flag build_pair --k 2 --d 9999 --metric params # 12514794 250295.88
echo "python preprocessing/data_generate.py --dataset nds --search_space Amoeba  --type normal --flag build_pair --k 2 --d 9999 --metric params # 12514794 250295.88" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space Amoeba --type reduce --flag build_pair --k 2 --d 9999 --metric params # 12514794 250295.88
echo "python preprocessing/data_generate.py --dataset nds --search_space Amoeba --type reduce --flag build_pair --k 2 --d 9999 --metric params # 12514794 250295.88" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3577690 71553.8
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3577690 71553.8" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d  --type normal --flag build_pair --k 2 --d 9999 --metric params  # 1293818 25876.36
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d  --type normal --flag build_pair --k 2 --d 9999 --metric params  # 1293818 25876.36" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space NASNet  --type normal --flag build_pair --k 2 --d 9999 --metric params # 12152490 243049.80000000002
echo "python preprocessing/data_generate.py --dataset nds --search_space NASNet  --type normal --flag build_pair --k 2 --d 9999 --metric params # 12152490 243049.80000000002" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space NASNet --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space NASNet --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3427242 68544.84
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3427242 68544.84" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space ENAS  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3685994 73719.88
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3685994 73719.88" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space ENAS --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space PNAS  --type normal --flag build_pair --k 2 --d 9999 --metric params # 12187242 243744.84
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS  --type normal --flag build_pair --k 2 --d 9999 --metric params # 12187242 243744.84" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space PNAS --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3427242 68544.84
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd  --type normal --flag build_pair --k 2 --d 9999 --metric params # 3427242 68544.84" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d  --type normal --flag build_pair --k 2 --d 9999 --metric params # 1024186 20483.72
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d  --type normal --flag build_pair --k 2 --d 9999 --metric params # 1024186 20483.72" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d --type reduce --flag build_pair --k 2 --d 9999 --metric params
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d --type reduce --flag build_pair --k 2 --d 9999 --metric params" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task class_scene # 41.443935 0.8288787000000001 (we multiply it by 100000, thus lower bound 82880.0)
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task class_scene # 41.443935 0.8288787000000001 (we multiply it by 100000, thus lower bound 82880.0)" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task class_object
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task class_object" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task autoencoder
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task autoencoder" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task normal
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task normal" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task jigsaw
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task jigsaw" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task room_layout
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task room_layout" >> commands_log.txt

python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task segmentsemantic
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 9999 --metric params --task segmentsemantic" >> commands_log.txt
