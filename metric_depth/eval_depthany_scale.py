import os
import pandas as pd

eval_list = ['/home/output/GrandTour/2024-11-02-17-10-25']
ckpt_path = '/home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth'
# depth_alignment = 'FALSE'
max_depth = 60


for depth_alignment in ['TRUE', 'FALSE']:
    metric_summary = {}
    csv_pattern = 'DepthAny_noAlignment_metric' if depth_alignment == 'FALSE' else 'DepthAny_Alignment_metric'

    for data_eval in eval_list:
        date_str = os.path.basename(data_eval)
        metric_summary[date_str] = {}

        for accumu_level in [1, 5, 25, 50, 100, 150]:
            txt_file = os.path.join(data_eval, f"accumulate_{str(accumu_level)}_pairs.txt")
            csv_file = txt_file.replace('pairs.txt', f'_2024-11-02-17-10-25_{csv_pattern}.csv')
            print(f"evaluating dataset: {data_eval} in accumulate level {str(accumu_level)}")
            os.system(f"python /home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/metric_depth/eval.py \
                        --pretrained_from {ckpt_path} --max_depth {max_depth} --depth_alignment {depth_alignment} \
                        --dataset_file_path {txt_file} --csv_file {csv_file}")
            
            csv_file = txt_file.replace('_pairs.txt', '_depth_anything_noAlignment_metric.csv')
            df = pd.read_csv(csv_file)
            metrics_dict = df.iloc[0].to_dict()

            metric_summary[date_str][accumu_level] = metrics_dict
            
    rows = []
    for date, accumu_data in metric_summary.items():
        for accumu_level, metrics in accumu_data.items():
            row = {
                'date': date,
                'accumulation_level': accumu_level,
                **metrics  # Unpack metrics into columns
            }
            rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Display as formatted table
    # =================================================================
    # Reorder columns (optional)
    cols = ['date', 'accumulation_level'] + [c for c in summary_df.columns if c not in ['date', 'accumulation_level']]
    summary_df = summary_df[cols]

    summary_df.to_csv(f"/home/output/2024-11-02-17-10-25_{csv_pattern}.csv", index=False)
            


