import os

scenes = ['Barn']
data_devices = ['cuda']

data_base_path="/tudelft.net/staff-umbrella/Deep3D/mingchiehhu/TNT_GOF/TrainingSet"
out_base_path='output_tnt'
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