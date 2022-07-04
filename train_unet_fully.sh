CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 25 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 50 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 100 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 150 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 200 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 250 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully --num_classes 2 --labeled_num 495 
