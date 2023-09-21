# ADJGIN
python universal_main.py --space nb201 --transfer_space nb101 --representation adj_gin --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space nb101 --transfer_space nb201 --representation adj_gin --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space ENAS --transfer_space DARTS --representation adj_gin --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space DARTS --transfer_space ENAS --representation adj_gin --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space PNAS_fix-w-d --transfer_space DARTS_fix-w-d --representation adj_gin --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space NASNet --transfer_space ENAS_fix-w-d --representation adj_gin --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4


python new_main.py --space nb101 --representation adj_gin --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space nb201 --representation adj_gin --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS --representation adj_gin --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space ENAS --representation adj_gin --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS_fix-w-d --representation adj_gin --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8  --transfer_comparison
python new_main.py --space ENAS_fix-w-d --representation adj_gin --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8   --transfer_comparison

# ADJGINZCP
python universal_main.py --space nb201 --transfer_space nb101 --representation adj_gin_zcp --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space nb101 --transfer_space nb201 --representation adj_gin_zcp --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space ENAS --transfer_space DARTS --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space DARTS --transfer_space ENAS --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space PNAS_fix-w-d --transfer_space DARTS_fix-w-d --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space NASNet --transfer_space ENAS_fix-w-d --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4


python new_main.py --space nb101 --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space nb201 --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space ENAS --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS_fix-w-d --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8  --transfer_comparison
python new_main.py --space ENAS_fix-w-d --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8   --transfer_comparison

# ADJGINCATE
python universal_main.py --space nb201 --transfer_space nb101 --representation adj_gin_cate --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space nb101 --transfer_space nb201 --representation adj_gin_cate --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space ENAS --transfer_space DARTS --representation adj_gin_cate --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space DARTS --transfer_space ENAS --representation adj_gin_cate --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space PNAS_fix-w-d --transfer_space DARTS_fix-w-d --representation adj_gin_cate --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space NASNet --transfer_space ENAS_fix-w-d --representation adj_gin_cate --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4


python new_main.py --space nb101 --representation adj_gin_cate --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space nb201 --representation adj_gin_cate --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS --representation adj_gin_cate --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space ENAS --representation adj_gin_cate --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS_fix-w-d --representation adj_gin_cate --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8  --transfer_comparison
python new_main.py --space ENAS_fix-w-d --representation adj_gin_cate --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8   --transfer_comparison

# ADJGINArch2Vec
python universal_main.py --space nb201 --transfer_space nb101 --representation adj_gin_arch2vec --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space nb101 --transfer_space nb201 --representation adj_gin_arch2vec --loss_type pwl --sample_size 1024 --batch_size 64 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space ENAS --transfer_space DARTS --representation adj_gin_arch2vec --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space DARTS --transfer_space ENAS --representation adj_gin_arch2vec --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space PNAS_fix-w-d --transfer_space DARTS_fix-w-d --representation adj_gin_arch2vec --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space NASNet --transfer_space ENAS_fix-w-d --representation adj_gin_arch2vec --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4


python new_main.py --space nb101 --representation adj_gin_arch2vec --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space nb201 --representation adj_gin_arch2vec --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS --representation adj_gin_arch2vec --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space ENAS --representation adj_gin_arch2vec --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS_fix-w-d --representation adj_gin_arch2vec --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8  --transfer_comparison
python new_main.py --space ENAS_fix-w-d --representation adj_gin_arch2vec --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8   --transfer_comparison

