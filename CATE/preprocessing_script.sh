####### PREPROCESSING #######
# !!!! ALL SS !!!!
python preprocessing/gen_alljson.py
# NASBench101
python preprocessing/gen_json.py
# NASBench201
python preprocessing/gen_json_nb201.py
# NASBench301
python preprocessing/gen_json_darts.py
# NDS Normal Cell
python preprocessing/gen_json_nds.py --type normal  --search_space Amoeba 
python preprocessing/gen_json_nds.py --type normal  --search_space PNAS_fix-w-d 
python preprocessing/gen_json_nds.py --type normal  --search_space ENAS_fix-w-d 
python preprocessing/gen_json_nds.py --type normal  --search_space NASNet 
python preprocessing/gen_json_nds.py --type normal  --search_space DARTS 
python preprocessing/gen_json_nds.py --type normal  --search_space ENAS 
python preprocessing/gen_json_nds.py --type normal  --search_space PNAS 
python preprocessing/gen_json_nds.py --type normal  --search_space DARTS_lr-wd 
python preprocessing/gen_json_nds.py --type normal  --search_space DARTS_fix-w-d 
# NDS Reduce Cell
python preprocessing/gen_json_nds.py --search_space Amoeba --type reduce
python preprocessing/gen_json_nds.py --search_space PNAS_fix-w-d --type reduce
python preprocessing/gen_json_nds.py --search_space ENAS_fix-w-d --type reduce
python preprocessing/gen_json_nds.py --search_space NASNet --type reduce
python preprocessing/gen_json_nds.py --search_space DARTS --type reduce
python preprocessing/gen_json_nds.py --search_space ENAS --type reduce
python preprocessing/gen_json_nds.py --search_space PNAS --type reduce
python preprocessing/gen_json_nds.py --search_space DARTS_lr-wd --type reduce
python preprocessing/gen_json_nds.py --search_space DARTS_fix-w-d --type reduce
# TransNASBench-101 Micro
python preprocessing/gen_json_transnasbench101.py  --task class_scene
python preprocessing/gen_json_transnasbench101.py  --task class_object
python preprocessing/gen_json_transnasbench101.py  --task autoencoder
python preprocessing/gen_json_transnasbench101.py  --task normal
python preprocessing/gen_json_transnasbench101.py  --task jigsaw
python preprocessing/gen_json_transnasbench101.py  --task room_layout
python preprocessing/gen_json_transnasbench101.py  --task segmentsemantic