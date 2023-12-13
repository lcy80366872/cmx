python main.py \
	--model CMMPNet \
        --batch_size 4 \
        --gpu_ids 1,2 \
        --epochs 50 \
        --dataset "TLCGIS" \
	--split_train_val_test "dataset/fusion_lidar_images_sigspatial18/" \
	--sat_dir "dataset/fusion_lidar_images_sigspatial18/rgb/" \
	--mask_dir "dataset/fusion_lidar_images_sigspatial18/mask/" \
	--lidar_dir "dataset/fusion_lidar_images_sigspatial18/depth_lpu/" \
        --weight_save_dir 'save_model/' \
        --lr 1e-4  
