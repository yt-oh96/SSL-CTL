FILE_NAME=$1
DEVICE_NUM=$2

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 25 --fold 0 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 50 --fold 0 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 100 --fold 0 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 150 --fold 0 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 200 --fold 0 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 250 --fold 0 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully0 --num_classes 2 --labeled_num 495 --fold 0 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 25 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 50 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 100 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 150 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 200 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 250 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully1 --num_classes 2 --labeled_num 495 --fold 1 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 25 --fold 2 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 50 --fold 2 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 100 --fold 2 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 150 --fold 2 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 200 --fold 2 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 250 --fold 2 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully2 --num_classes 2 --labeled_num 495 --fold 2 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 25 --fold 3 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 50 --fold 3 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 100 --fold 3 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 150 --fold 3 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 200 --fold 3 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 250 --fold 3 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully3 --num_classes 2 --labeled_num 495 --fold 3 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 25 --fold 4 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 50 --fold 4 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 100 --fold 4 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 150 --fold 4 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 200 --fold 4 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 250 --fold 4 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully4 --num_classes 2 --labeled_num 495 --fold 4 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 25 --fold 5 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 50 --fold 5 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 100 --fold 5 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 150 --fold 5 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 200 --fold 5 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 250 --fold 5 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully5 --num_classes 2 --labeled_num 495 --fold 5 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 25 --fold 6 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 50 --fold 6 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 100 --fold 6 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 150 --fold 6 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 200 --fold 6 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 250 --fold 6 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully6 --num_classes 2 --labeled_num 495 --fold 6 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 25 --fold 7 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 50 --fold 7 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 100 --fold 7 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 150 --fold 7 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 200 --fold 7 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 250 --fold 7 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully7 --num_classes 2 --labeled_num 495 --fold 7 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 25 --fold 8 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 50 --fold 8 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 100 --fold 8 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 150 --fold 8 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 200 --fold 8 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 250 --fold 8 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully8 --num_classes 2 --labeled_num 495 --fold 8 && \

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 25 --fold 9 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 50 --fold 9 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 100 --fold 9 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 150 --fold 9 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 200 --fold 9 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 250 --fold 9 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp UNet_fully9 --num_classes 2 --labeled_num 495 --fold 9
