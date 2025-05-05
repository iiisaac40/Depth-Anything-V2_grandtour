#!/bin/bash
# Usage: ./train.sh [--epoch NUM] [--bs SIZE] [--gpus NUM] [--lr RATE] [--encoder TYPE] 
#                   [--dataset DS] [--img-size SIZE] [--min-depth MIN] [--max-depth MAX]
#                   [--pretrained-from PATH] [--save-path DIR] [--accumulation LEVEL]
#                   [--train-txt-file PATH] [--val-txt-file PATH]

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epoch)
            epoch="$2"
            shift; shift ;;
        --bs)
            bs="$2"
            shift; shift ;;
        --gpus)
            gpus="$2"
            shift; shift ;;
        --lr)
            lr="$2"
            shift; shift ;;
        --encoder)
            encoder="$2"
            shift; shift ;;
        --dataset)
            dataset="$2"
            shift; shift ;;
        --img-size)
            img_size="$2"
            shift; shift ;;
        --min-depth)
            min_depth="$2"
            shift; shift ;;
        --max-depth)
            max_depth="$2"
            shift; shift ;;
        --pretrained-from)
            pretrained_from="$2"
            shift; shift ;;
        --save-path)
            save_path="$2"
            shift; shift ;;
        --train-txt-file)
            train_txt_file="$2"
            shift; shift ;;
        --val-txt-file)
            val_txt_file="$2"
            shift; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --epoch NUM              Number of epochs (default: 120)"
            echo "  --bs SIZE                Batch size (default: 4)"
            echo "  --gpus NUM               Number of GPUs (default: 8)"
            echo "  --lr RATE                Learning rate (default: 0.000005)"
            echo "  --encoder TYPE           Encoder type (default: vitl)"
            echo "  --dataset DS             Dataset name (default: grandtour)"
            echo "  --img-size SIZE          Image size (default: 518)"
            echo "  --min-depth MIN          Minimum depth (default: 0.001)"
            echo "  --max-depth MAX          Maximum depth (default: 60)"
            echo "  --pretrained-from PATH   Pretrained weights path"
            echo "  --save-path DIR          Output directory (default: exp2/grandtour)"
            echo "  --train-txt-file PATH    Training text file path"
            echo "  --val-txt-file PATH      Validation text file path"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# Set default values
now=$(date +"%Y%m%d_%H%M%S")
epoch=${epoch:-120}
bs=${bs:-2}
gpus=${gpus:-8}
lr=${lr:-0.000005}
encoder=${encoder:-vitl}
dataset=${dataset:-grandtour}
img_size=${img_size:-518}
min_depth=${min_depth:-0.001}
max_depth=${max_depth:-40}
pretrained_from=${pretrained_from:-/home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth}
save_path=${save_path:-exp2/grandtour}
train_txt_file=${train_txt_file:-NotSet}
val_txt_file=${val_txt_file:-NotSet}

MASTER_PORT=$(shuf -i 20000-65000 -n 1)
mkdir -p "$save_path"

# Training command
python3 -m torch.distributed.launch \
    --nproc_per_node="$gpus" \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    train.py \
    --epoch "$epoch" \
    --encoder "$encoder" \
    --bs "$bs" \
    --lr "$lr" \
    --save-path "$save_path" \
    --dataset "$dataset" \
    --img-size "$img_size" \
    --train-txt-file "$train_txt_file" \
    --val-txt-file "$val_txt_file" \
    --min-depth "$min_depth" \
    --max-depth "$max_depth" \
    --pretrained-from "$pretrained_from" 2>&1 | tee -a "$save_path/$now.log"