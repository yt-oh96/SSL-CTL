DEVICE_NUM=$1
CASE=$2

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train_transform_Label_contrastive_cross_pseudo_supervision_ablation.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_cps_albation_case$CASE --TDM True --Local_Contrastive False --Transform False --num_classes 2 --labeled_num 25 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train_transform_Label_contrastive_cross_pseudo_supervision_ablation.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_cps_albation_case$CASE --TDM True --Local_Contrastive False --Transform False --num_classes 2 --labeled_num 50 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train_transform_Label_contrastive_cross_pseudo_supervision_ablation.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_cps_albation_case$CASE --TDM True --Local_Contrastive False --Transform False --num_classes 2 --labeled_num 100 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train_transform_Label_contrastive_cross_pseudo_supervision_ablation.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_cps_albation_case$CASE --TDM True --Local_Contrastive False --Transform False --num_classes 2 --labeled_num 150 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train_transform_Label_contrastive_cross_pseudo_supervision_ablation.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_cps_albation_case$CASE --TDM True --Local_Contrastive False --Transform False --num_classes 2 --labeled_num 200 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train_transform_Label_contrastive_cross_pseudo_supervision_ablation.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp transform_Label_contrastive_cps_albation_case$CASE --TDM True --Local_Contrastive False --Transform False --num_classes 2 --labeled_num 250 --fold 1 
