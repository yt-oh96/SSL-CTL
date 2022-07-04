FILE_NAME=$1
DEVICE_NUM=$2

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Mean_Teacher_fold1 --num_classes 2 --labeled_num 25 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Mean_Teacher_fold1 --num_classes 2 --labeled_num 50 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Mean_Teacher_fold1 --num_classes 2 --labeled_num 100 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Mean_Teacher_fold1 --num_classes 2 --labeled_num 150 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Mean_Teacher_fold1 --num_classes 2 --labeled_num 200 --fold 1 && \
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python $FILE_NAME --root_path /notebook/SSL4MIS/data/UFMR_DCMR --exp Mean_Teacher_fold1 --num_classes 2 --labeled_num 250 --fold 1 
