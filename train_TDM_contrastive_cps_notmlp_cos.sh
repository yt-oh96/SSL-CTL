CUDA_VISIBLE_DEVICES=0 python train_transform_TDM_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_TDM_contrastive_mlp_cross_pseudo_notmlp_cos --num_classes 2 --labeled_num 50 && \
CUDA_VISIBLE_DEVICES=0 python train_transform_TDM_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_TDM_contrastive_mlp_cross_pseudo_notmlp_cos --num_classes 2 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python train_transform_TDM_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_TDM_contrastive_mlp_cross_pseudo_notmlp_cos --num_classes 2 --labeled_num 150 && \
CUDA_VISIBLE_DEVICES=0 python train_transform_TDM_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_TDM_contrastive_mlp_cross_pseudo_notmlp_cos --num_classes 2 --labeled_num 200 && \
CUDA_VISIBLE_DEVICES=0 python train_transform_TDM_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_TDM_contrastive_mlp_cross_pseudo_notmlp_cos --num_classes 2 --labeled_num 250 