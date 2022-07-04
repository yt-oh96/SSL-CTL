FILE_NAME=$1

CUDA_VISIBLE_DEVICES=1 python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold0 --num_classes 2 --labeled_num 25 --fold 0 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold0 --num_classes 2 --labeled_num 50 --fold 0 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold0 --num_classes 2 --labeled_num 100 --fold 0 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold0 --num_classes 2 --labeled_num 150 --fold 0 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold0 --num_classes 2 --labeled_num 200 --fold 0 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold0 --num_classes 2 --labeled_num 250 --fold 0 && \

CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold1 --num_classes 2 --labeled_num 25 --fold 1 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold1 --num_classes 2 --labeled_num 50 --fold 1 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold1 --num_classes 2 --labeled_num 100 --fold 1 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold1 --num_classes 2 --labeled_num 150 --fold 1 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold1 --num_classes 2 --labeled_num 200 --fold 1 && \
CUDA_VISIBLE_DEVICES=1 python train_mean_teacher_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold1 --num_classes 2 --labeled_num 250 --fold 1 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold2 --num_classes 2 --labeled_num 25 --fold 2 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold2 --num_classes 2 --labeled_num 50 --fold 2 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold2 --num_classes 2 --labeled_num 100 --fold 2 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold2 --num_classes 2 --labeled_num 150 --fold 2 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold2 --num_classes 2 --labeled_num 200 --fold 2 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold2 --num_classes 2 --labeled_num 250 --fold 2 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold3 --num_classes 2 --labeled_num 25 --fold 3 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold3 --num_classes 2 --labeled_num 50 --fold 3 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold3 --num_classes 2 --labeled_num 100 --fold 3 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold3 --num_classes 2 --labeled_num 150 --fold 3 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold3 --num_classes 2 --labeled_num 200 --fold 3 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold3 --num_classes 2 --labeled_num 250 --fold 3 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold4 --num_classes 2 --labeled_num 25 --fold 4 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold4 --num_classes 2 --labeled_num 50 --fold 4 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold4 --num_classes 2 --labeled_num 100 --fold 4 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold4 --num_classes 2 --labeled_num 150 --fold 4 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold4 --num_classes 2 --labeled_num 200 --fold 4 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold4 --num_classes 2 --labeled_num 250 --fold 4 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold5 --num_classes 2 --labeled_num 25 --fold 5 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold5 --num_classes 2 --labeled_num 50 --fold 5 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold5 --num_classes 2 --labeled_num 100 --fold 5 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold5 --num_classes 2 --labeled_num 150 --fold 5 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold5 --num_classes 2 --labeled_num 200 --fold 5 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold5 --num_classes 2 --labeled_num 250 --fold 5 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold6 --num_classes 2 --labeled_num 25 --fold 6 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold6 --num_classes 2 --labeled_num 50 --fold 6 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold6 --num_classes 2 --labeled_num 100 --fold 6 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold6 --num_classes 2 --labeled_num 150 --fold 6 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold6 --num_classes 2 --labeled_num 200 --fold 6 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold6 --num_classes 2 --labeled_num 250 --fold 6 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold7 --num_classes 2 --labeled_num 25 --fold 7 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold7 --num_classes 2 --labeled_num 50 --fold 7 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold7 --num_classes 2 --labeled_num 100 --fold 7 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold7 --num_classes 2 --labeled_num 150 --fold 7 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold7 --num_classes 2 --labeled_num 200 --fold 7 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold7 --num_classes 2 --labeled_num 250 --fold 7 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold8 --num_classes 2 --labeled_num 25 --fold 8 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold8 --num_classes 2 --labeled_num 50 --fold 8 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold8 --num_classes 2 --labeled_num 100 --fold 8 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold8 --num_classes 2 --labeled_num 150 --fold 8 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold8 --num_classes 2 --labeled_num 200 --fold 8 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold8 --num_classes 2 --labeled_num 250 --fold 8 && \

CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold9 --num_classes 2 --labeled_num 25 --fold 9 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold9 --num_classes 2 --labeled_num 50 --fold 9 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold9 --num_classes 2 --labeled_num 100 --fold 9 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold9 --num_classes 2 --labeled_num 150 --fold 9 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold9 --num_classes 2 --labeled_num 200 --fold 9 && \
CUDA_VISIBLE_DEVICES=1 python train_transform_Label_contrastive_cross_pseudo_supervision_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold9 --num_classes 2 --labeled_num 250 --fold 9