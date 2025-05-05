import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yaml


SCHEDULE = True
submit_dir ="/cluster/home/haozhu1/Thesis/result/.submit_depth_training"

if not os.path.exists(submit_dir):
    os.makedirs(submit_dir)


batch_size = 4
lr = 5e-6

txt_file = '/mnt/txt_files/'

for max_depth in [80, 60, 40, 20]:
    for accum_frames in [1, 5, 25, 50, 100, 150]:
        train_txt = os.path.join(txt_file, f'train_{str(accum_frames)}_files.txt')
        val_txt = os.path.join(txt_file, f'val_{str(accum_frames)}_files.txt')
    
        content = f"""#!/bin/bash
    
#SBATCH --account=es_hutter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --gres=gpumem:26000m
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8888
#SBATCH --tmp=200000
#SBATCH --output="/cluster/home/haozhu1/Thesis/.out/depth_training_accum{accum_frames}_{max_depth}_output.log"
#SBATCH --error="/cluster/home/haozhu1/Thesis/.out/depth_training_accum{accum_frames}_{max_depth}_err.log"
#SBATCH --open-mode=truncate
export WANDB_API_KEY=f795cedf7f928832139004e8d5fc078722adb89d
mkdir -p $TMPDIR/GrandTour
tar -xf /cluster/scratch/haozhu1/Thesis/container/grandtour_depth_benchmark2.tar -C $TMPDIR
tar -xf /cluster/scratch/haozhu1/depth_data/updated_images/GrandTour/GrandTour.tar  -C $TMPDIR/GrandTour
# tar -xf /cluster/scratch/haozhu1/depth_data/updated_images/noOcclude/GrandTour.tar  -C $TMPDIR/GrandTour

module load stack/2024-04 gcc/8.5.0 cuda/12.1.1 eth_proxy

apptainer exec --nv --containall --writable --env WANDB_API_KEY=$WANDB_API_KEY --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu  \
--env HF_HOME=/mnt/.cache/huggingface \
--env TRANSFORMERS_CACHE=/mnt/.cache/huggingface \
--env XDG_CACHE_HOME=/mnt/.cache \
--env MPLCONFIGDIR=/mnt/.config/matplotlib \
--bind $TMPDIR/GrandTour/:/mnt/ \
--bind /cluster/scratch/haozhu1/Depth-Anything-V2_grandtour/metric_depth:/home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth \
--bind /cluster/home/haozhu1/Thesis/gits/Depth-Anything-V2_grandtour/metric_depth/dataset/grandtour.py:/home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/dataset/grandtour.py \
--bind /cluster/home/haozhu1/Thesis/gits/Depth-Anything-V2_grandtour/metric_depth/train.py:/home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/train.py \
$TMPDIR/grandtour_depth_benchmark2.sif \
/bin/bash -c " \
export HOME=/home && \
source /opt/conda/etc/profile.d/conda.sh && \
conda activate grandtour && ls /mnt && \
source /opt/ros/noetic/setup.bash && \
source /home/grand_tour_depth_benchmark/catkin_ws/devel/setup.bash && \
pip install wandb && wandb login $WANDB_API_KEY && \
cd /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth && \
python -c 'import torch; print(torch.cuda.device_count())' && \
bash /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/dist_train.sh \
    --epoch 20 --save-path exp{max_depth}/grandtour{accum_frames} \
    --val-txt-file {val_txt} --train-txt-file {train_txt} --bs 4 --max-depth {max_depth} && \

"


exit 0
            """
    
        # bash /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/dist_train.sh
    
        script_path = os.path.join(submit_dir, "depth_training.sh")
        with open(script_path, "w") as file:
            file.write(content)
    
        if SCHEDULE:
            os.system(f"sbatch {script_path}")