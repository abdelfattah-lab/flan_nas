# ####### DATA GENERATION #######
# # !!!! ALL SS !!!!
# python preprocessing/data_generate.py --dataset all_ss --flag extract_seq
# echo "python preprocessing/data_generate.py --dataset all_ss --flag extract_seq" >> commands_log.txt


# # NASBench101
# python preprocessing/data_generate.py --dataset nasbench101 --flag extract_seq
# echo "python preprocessing/data_generate.py --dataset nasbench101 --flag extract_seq" >> commands_log.txt


# NASBench201
python preprocessing/data_generate.py --dataset nasbench201 --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nasbench201 --flag extract_seq" >> commands_log.txt


# NASBench301
python preprocessing/data_generate.py --dataset nasbench301 --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nasbench301 --flag extract_seq" >> commands_log.txt


# NASBench301 NR
python preprocessing/data_generate.py --dataset nb301a --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nb301a --flag extract_seq" >> commands_log.txt
python preprocessing/data_generate.py --dataset nb301b --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nb301b --flag extract_seq" >> commands_log.txt

# NDS
python preprocessing/data_generate.py --dataset nds --search_space Amoeba --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space Amoeba --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space Amoeba  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space Amoeba  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space NASNet --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space NASNet --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space NASNet  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space NASNet  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space DARTS --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space ENAS --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space ENAS  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space ENAS  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space PNAS --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space PNAS  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space PNAS  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd  --type reduce --flag extract_seq" >> commands_log.txt


python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d --type normal --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d --type normal --flag extract_seq" >> commands_log.txt

python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d  --type reduce --flag extract_seq
echo "python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d  --type reduce --flag extract_seq" >> commands_log.txt


# TransNASBench-101 Micro
python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task class_scene
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task class_scene" >> commands_log.txt


python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task class_object
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task class_object" >> commands_log.txt


python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task autoencoder
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task autoencoder" >> commands_log.txt


python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task normal
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task normal" >> commands_log.txt


python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task jigsaw
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task jigsaw" >> commands_log.txt


python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task room_layout
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task room_layout" >> commands_log.txt


python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task segmentsemantic
echo "python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task segmentsemantic" >> commands_log.txt

