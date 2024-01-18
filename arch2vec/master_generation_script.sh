####### PREPROCESSING #######

# !!!! ALL SS !!!!
python preprocessing/gen_alljson.py     
echo "python preprocessing/gen_alljson.py     " >> commands_log.txt

# NASBench101
python preprocessing/gen_json.py             
echo "python preprocessing/gen_json.py             " >> commands_log.txt
# NASBench201
python preprocessing/nasbench201_json.py     
echo "python preprocessing/nasbench201_json.py     " >> commands_log.txt
# NASBench301
python preprocessing/nasbench301_json.py     
echo "python preprocessing/nasbench301_json.py     " >> commands_log.txt

# NDS Normal Cell
python preprocessing/nds_json.py --search_space Amoeba --type normal 
echo "python preprocessing/nds_json.py --search_space Amoeba --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space PNAS_fix-w-d --type normal 
echo "python preprocessing/nds_json.py --search_space PNAS_fix-w-d --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space ENAS_fix-w-d --type normal 
echo "python preprocessing/nds_json.py --search_space ENAS_fix-w-d --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space NASNet --type normal 
echo "python preprocessing/nds_json.py --search_space NASNet --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space DARTS --type normal 
echo "python preprocessing/nds_json.py --search_space DARTS --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space ENAS --type normal 
echo "python preprocessing/nds_json.py --search_space ENAS --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space PNAS --type normal 
echo "python preprocessing/nds_json.py --search_space PNAS --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space DARTS_lr-wd --type normal 
echo "python preprocessing/nds_json.py --search_space DARTS_lr-wd --type normal " >> commands_log.txt
python preprocessing/nds_json.py --search_space DARTS_fix-w-d --type normal 
echo "python preprocessing/nds_json.py --search_space DARTS_fix-w-d --type normal " >> commands_log.txt

# NDS Reduce Cell
python preprocessing/nds_json.py --search_space Amoeba --type reduce
echo "python preprocessing/nds_json.py --search_space Amoeba --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space PNAS_fix-w-d --type reduce
echo "python preprocessing/nds_json.py --search_space PNAS_fix-w-d --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space ENAS_fix-w-d --type reduce
echo "python preprocessing/nds_json.py --search_space ENAS_fix-w-d --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space NASNet --type reduce
echo "python preprocessing/nds_json.py --search_space NASNet --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space DARTS --type reduce
echo "python preprocessing/nds_json.py --search_space DARTS --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space ENAS --type reduce
echo "python preprocessing/nds_json.py --search_space ENAS --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space PNAS --type reduce
echo "python preprocessing/nds_json.py --search_space PNAS --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space DARTS_lr-wd --type reduce
echo "python preprocessing/nds_json.py --search_space DARTS_lr-wd --type reduce" >> commands_log.txt
python preprocessing/nds_json.py --search_space DARTS_fix-w-d --type reduce
echo "python preprocessing/nds_json.py --search_space DARTS_fix-w-d --type reduce" >> commands_log.txt

# TransNASBench-101 Micro
python preprocessing/transnasbench101_json.py --task class_scene
echo "python preprocessing/transnasbench101_json.py --task class_scene" >> commands_log.txt
python preprocessing/transnasbench101_json.py --task class_object
echo "python preprocessing/transnasbench101_json.py --task class_object" >> commands_log.txt
python preprocessing/transnasbench101_json.py --task autoencoder
echo "python preprocessing/transnasbench101_json.py --task autoencoder" >> commands_log.txt
python preprocessing/transnasbench101_json.py --task normal
echo "python preprocessing/transnasbench101_json.py --task normal" >> commands_log.txt
python preprocessing/transnasbench101_json.py --task jigsaw
echo "python preprocessing/transnasbench101_json.py --task jigsaw" >> commands_log.txt
python preprocessing/transnasbench101_json.py --task room_layout
echo "python preprocessing/transnasbench101_json.py --task room_layout" >> commands_log.txt
python preprocessing/transnasbench101_json.py --task segmentsemantic
echo "python preprocessing/transnasbench101_json.py --task segmentsemantic" >> commands_log.txt

####### PRETRAINING #######

# !!!! ALL SS !!!!
python -i models/pretraining_allss.py --input_dim 20 --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name all_ss 
echo "python -i models/pretraining_allss.py --input_dim 20 --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name all_ss " >> commands_log.txt

# NASBench101
python -i  models/pretraining_nasbench101.py  --input_dim 5 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nasbench101
echo "python -i  models/pretraining_nasbench101.py  --input_dim 5 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nasbench101" >> commands_log.txt
# NASBench201
python -i models/pretraining_nasbench201.py  --input_dim 7 --hops 5 --epochs 10 --bs 32 --cfg 4 --seed 4 --name nasbench201
echo "python -i models/pretraining_nasbench201.py  --input_dim 7 --hops 5 --epochs 10 --bs 32 --cfg 4 --seed 4 --name nasbench201" >> commands_log.txt
# NASBench301
python -i models/pretraining_nasbench301.py  --input_dim 11 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nasbench301
echo "python -i models/pretraining_nasbench301.py  --input_dim 11 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nasbench301" >> commands_log.txt
# NDS Normal Cell
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space Amoeba --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space Amoeba --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS_fix-w-d --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS_fix-w-d --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS_fix-w-d --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS_fix-w-d --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space NASNet --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space NASNet --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_lr-wd --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_lr-wd --type normal" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_fix-w-d --type normal
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_fix-w-d --type normal" >> commands_log.txt
# NDS Reduce Cell
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space Amoeba --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space Amoeba --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS_fix-w-d --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS_fix-w-d --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS_fix-w-d --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS_fix-w-d --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space NASNet --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space NASNet --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space ENAS --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space PNAS --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_lr-wd --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_lr-wd --type reduce" >> commands_log.txt
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_fix-w-d --type reduce
echo "python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 10 --seed 1 --name nds --search_space DARTS_fix-w-d --type reduce" >> commands_log.txt
# TransNASBench-101 Micro
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task class_scene
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task class_scene" >> commands_log.txt
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task class_object
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task class_object" >> commands_log.txt
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task autoencoder
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task autoencoder" >> commands_log.txt
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task normal
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task normal" >> commands_log.txt
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task jigsaw
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task jigsaw" >> commands_log.txt
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task room_layout
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task room_layout" >> commands_log.txt
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task segmentsemantic
echo "python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 10 --seed 1 --name tb101 --task segmentsemantic" >> commands_log.txt

####### Arch2Vec Extraction #######

# !!!! ALL SS !!!!
bash run_scripts/extract_allss.sh
echo "bash run_scripts/extract_allss.sh" >> commands_log.txt

# NASBench101
bash run_scripts/extract_arch2vec.sh
echo "bash run_scripts/extract_arch2vec.sh" >> commands_log.txt
# NASBench201
bash run_scripts/extract_arch2vec_nasbench201.sh
echo "bash run_scripts/extract_arch2vec_nasbench201.sh" >> commands_log.txt
# NASBench301
bash run_scripts/extract_arch2vec_nasbench301.sh
echo "bash run_scripts/extract_arch2vec_nasbench301.sh" >> commands_log.txt
# NASBench301 NR
bash run_scripts/extract_arch2vec_nb301_a_b.sh
echo "bash run_scripts/extract_arch2vec_nb301_a_b.sh" >> commands_log.txt
# NDS
bash run_scripts/extract_arch2vec_nds.sh
echo "bash run_scripts/extract_arch2vec_nds.sh" >> commands_log.txt
# TransNASBench-101 Micro
bash run_scripts/extract_arch2vec_transnasbench101.sh
echo "bash run_scripts/extract_arch2vec_transnasbench101.sh" >> commands_log.txt