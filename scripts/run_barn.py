import os

scenes = ['Barn']
data_devices = ['cuda']

data_base_path="/tudelft.net/staff-umbrella/Deep3D/mingchiehhu/TNT_GOF/TrainingSet"
out_base_path='output_tnt'
# out_base_path='output_tnt_highres'
out_name='test'
gpu_id=0

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)
    
    common_args = f"--quiet -r2 --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --opacity_cull_threshold 0.05 --exposure_compensation"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    # if scene in ['Meetingroom', 'Courthouse']:
    #     depth_trunc = 4.5
    #     sdf_trunc = 0.024
    # else:
    #     depth_trunc = 3
    #     sdf_trunc = 0.016
    
    # common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter --voxel_size 0.004 --max_depth {depth_trunc} --sdf_trunc {sdf_trunc} --skip_test"
    common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter --voxel_size 0.004 --skip_test"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)