CUDA_VISIBLE_DEVICES=3 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_mlp_cross_pseudo_notmlp_euclidean_com_s --num_classes 2 --labeled_num 25 && \
CUDA_VISIBLE_DEVICES=3 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_mlp_cross_pseudo_notmlp_euclidean_com_s --num_classes 2 --labeled_num 50 && \
CUDA_VISIBLE_DEVICES=3 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_mlp_cross_pseudo_notmlp_euclidean_com_s --num_classes 2 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=3 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_mlp_cross_pseudo_notmlp_euclidean_com_s --num_classes 2 --labeled_num 150 && \
CUDA_VISIBLE_DEVICES=3 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_mlp_cross_pseudo_notmlp_euclidean_com_s --num_classes 2 --labeled_num 200 && \
CUDA_VISIBLE_DEVICES=3 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_mlp_cross_pseudo_notmlp_euclidean_com_s --num_classes 2 --labeled_num 250 
