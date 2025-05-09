import os
import glob
import random

SCHEDULE = True
submit_dir ="/cluster/home/haozhu1/Thesis/result/.eval_depthany"
# ckpt: /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/exp{max_depth}/grandtour{str(accum_frames)}/latest.pth
# ckpt: /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
if not os.path.exists(submit_dir):
    os.makedirs(submit_dir)


depth_alignment = 'FALSE'

txt_file = '/mnt/txt_files/'
for max_depth in [80, 60, 40, 20]:
    for accum_frames in [1, 5, 25, 50, 100, 150]:
        csv_pattern = f'depthAny_accum{str(accum_frames)}_maxdepth{max_depth}'
    
        test_txt = os.path.join(txt_file, f'test_1_files.txt')
        csv_file = test_txt.replace('files.txt', f'{csv_pattern}.csv')
    
        # test_txt = os.path.join('/mnt/KITTI', "val_pairs.txt")
        # csv_file = test_txt.replace('pairs.txt', f'{csv_pattern}.csv')
    
    
        master_port = random.randint(10000, 20000)
        content = f"""#!/bin/bash

#SBATCH --account=es_hutter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12288m
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=13312
#SBATCH --tmp=90000
#SBATCH --output="/cluster/home/haozhu1/Thesis/.out/depthany_eval_{str(accum_frames)}_depth{str(max_depth)}_out.log"
#SBATCH --error="/cluster/home/haozhu1/Thesis/.out/depthany_eval_{str(accum_frames)}_depth{str(max_depth)}_out.log"
#SBATCH --open-mode=truncate

mkdir -p $TMPDIR/GrandTour
tar -xf /cluster/scratch/haozhu1/Thesis/container/grandtour_depth_benchmark2.tar -C $TMPDIR
tar -xf /cluster/scratch/haozhu1/depth_data/updated_images/GrandTour/GrandTour.tar  -C $TMPDIR/GrandTour

# tar -xf /cluster/scratch/haozhu1/depth_data/eval_images/KITTI.tar -C $TMPDIR/GrandTour
# sed -i 's|/cluster/scratch/haozhu1/depth_data/eval_images|/mnt|g' $TMPDIR/GrandTour/KITTI/val_pairs.txt

module load stack/2024-04 gcc/8.5.0 cuda/12.1.1 eth_proxy

apptainer exec --nv --containall --writable --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu  \
  --env HF_HOME=/mnt/.cache/huggingface \
  --env TRANSFORMERS_CACHE=/mnt/.cache/huggingface \
  --env XDG_CACHE_HOME=/mnt/.cache \
  --env MPLCONFIGDIR=/mnt/.config/matplotlib \
  --bind $TMPDIR/GrandTour:/mnt/ \
  $TMPDIR/grandtour_depth_benchmark2.sif \
  /bin/bash -c "
  export HOME=/home && export KLEINKRAM_ACTIVE=ACTIVE && \
  source /opt/conda/etc/profile.d/conda.sh && \
  conda activate grandtour && \
  source /opt/ros/noetic/setup.bash && \
  source /home/grand_tour_depth_benchmark/catkin_ws/devel/setup.bash && \
  python -c 'import torch; print(torch.cuda.device_count())' && \
  ls /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/ && \
  python /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/eval.py \
  --pretrained_from /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/exp{max_depth}/grandtour{str(accum_frames)}/latest.pth    \
  --max_depth {max_depth} --depth_alignment {depth_alignment} \
  --dataset_txt_path {test_txt} --dataset_root_dir /mnt/GrandTour/  \
  --csv_file {csv_file}  --port {master_port} --dataset grandtour --vis_res FALSE
  "

  cp -r $TMPDIR/GrandTour/*/*.csv /cluster/scratch/haozhu1/depth_data/updated_images
  cp -r $TMPDIR/GrandTour/*/visualizations /cluster/scratch/haozhu1/depth_data/updated_images
 
  
exit 0
        """
# --vis_res TRUE
        script_path = os.path.join(submit_dir, f"eval_depthany_accum{accum_frames}.sh")
        with open(script_path, "w") as file:
            file.write(content)
    
        if SCHEDULE:
            os.system(f"sbatch {script_path}")



