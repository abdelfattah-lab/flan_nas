
####### PRETRAINING #######

# !!!! ALL SS !!!!
bash run_scripts/pretrain_allss.sh
echo "bash run_scripts/pretrain_allss.sh" >> commands_log.txt

bash run_scripts/pretrain_nasbench101.sh
echo "bash run_scripts/pretrain_nasbench101.sh" >> commands_log.txt

bash run_scripts/pretrain_nasbench201.sh
echo "bash run_scripts/pretrain_nasbench201.sh" >> commands_log.txt

bash run_scripts/pretrain_nasbench301.sh
echo "bash run_scripts/pretrain_nasbench301.sh" >> commands_log.txt

bash run_scripts/pretrain_nds.sh
echo "bash run_scripts/pretrain_nds.sh" >> commands_log.txt

bash run_scripts/pretrain_transnasbench101.sh
echo "bash run_scripts/pretrain_transnasbench101.sh" >> commands_log.txt

