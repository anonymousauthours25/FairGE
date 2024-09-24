#!/bin/bash



# facebook
python train_FairGE.py --dataset facebook --sens_attr 'region' --drop_feat_rate 0.1 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 2  --ge_alpha 0.1 --ge_beta 0.1
python train_FairGE.py --dataset facebook --sens_attr 'region' --drop_feat_rate 0.2 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 2  --ge_alpha 0.2 --ge_beta 0.1
python train_FairGE.py --dataset facebook --sens_attr 'region' --drop_feat_rate 0.3 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 10 --ge_alpha 0.1 --ge_beta 0.1
python train_FairGE.py --dataset facebook --sens_attr 'region' --drop_feat_rate 0.4 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 5  --ge_alpha 0.2 --ge_beta 0.1
python train_FairGE.py --dataset facebook --sens_attr 'region' --drop_feat_rate 0.5 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 5  --ge_alpha 0.1 --ge_beta 0.2
python train_FairGE.py --dataset facebook --sens_attr 'region' --drop_feat_rate 0.6 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 2  --ge_alpha 0.1 --ge_beta 0.1

# pokec-z-R
python train_FairGE.py --dataset poekc_z  --sens_attr 'region' --drop_feat_rate 0.1 --num_hidden 128 --nheads 1  --imp_method 'zero' --pe_dim 10 --ge_alpha 0.1 --ge_beta 0.2
python train_FairGE.py --dataset poekc_z  --sens_attr 'region' --drop_feat_rate 0.2 --num_hidden 32  --nheads 1  --imp_method 'zero' --pe_dim 2  --ge_alpha 0.2 --ge_beta 0.2
python train_FairGE.py --dataset poekc_z  --sens_attr 'region' --drop_feat_rate 0.3 --num_hidden 128 --nheads 1  --imp_method 'zero' --pe_dim 10 --ge_alpha 0.2 --ge_beta 0.1
python train_FairGE.py --dataset poekc_z  --sens_attr 'region' --drop_feat_rate 0.4 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 5  --ge_alpha 0.2 --ge_beta 0.2
python train_FairGE.py --dataset poekc_z  --sens_attr 'region' --drop_feat_rate 0.5 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 2  --ge_alpha 0.2 --ge_beta 0.1
python train_FairGE.py --dataset poekc_z  --sens_attr 'region' --drop_feat_rate 0.6 --num_hidden 64  --nheads 1  --imp_method 'zero' --pe_dim 10 --ge_alpha 0.1 --ge_beta 0.1


