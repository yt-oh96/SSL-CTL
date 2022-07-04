CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp TDM_cross_pseudo_10 --num_classes 2 --labeled_num 50 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp TDM_cross_pseudo_20 --num_classes 2 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp TDM_cross_pseudo_30 --num_classes 2 --labeled_num 150 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp TDM_cross_pseudo_40 --num_classes 2 --labeled_num 200 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp TDM_cross_pseudo_50 --num_classes 2 --labeled_num 250