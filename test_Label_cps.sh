MODE=$1
FOLD=$2

CUDA_VISIBLE_DEVICES=0 python test_tdm_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold$FOLD --model Label_contrastive_UNet --num_classes 2 --labeled_num 25 --mode $MODE && \
CUDA_VISIBLE_DEVICES=0 python test_tdm_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold$FOLD --model Label_contrastive_UNet --num_classes 2 --labeled_num 50 --mode $MODE && \
CUDA_VISIBLE_DEVICES=0 python test_tdm_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold$FOLD --model Label_contrastive_UNet --num_classes 2 --labeled_num 100 --mode $MODE && \
CUDA_VISIBLE_DEVICES=0 python test_tdm_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold$FOLD --model Label_contrastive_UNet --num_classes 2 --labeled_num 150 --mode $MODE && \
CUDA_VISIBLE_DEVICES=0 python test_tdm_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold$FOLD --model Label_contrastive_UNet --num_classes 2 --labeled_num 200 --mode $MODE && \
CUDA_VISIBLE_DEVICES=0 python test_tdm_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Label_cps_fold$FOLD --model Label_contrastive_UNet --num_classes 2 --labeled_num 250 --mode $MODE
