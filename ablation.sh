#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_LOG="${OUTPUT_LOG:-$SCRIPT_DIR/output.log}"

# Define the data types and model types
data_types=("epts" "mri" "epts,mri")
# data_types=("clinical")
model_types=("logistic" "random_forest" "lgbm" "balanced_random_forest")


# Loop through each data type and model type and run the command
for data in "${data_types[@]}"; do
    for model in "${model_types[@]}"; do
        for sum_epts in "" "--sum_epts"; do
            for calibrate in "" "--calibrate"; do
                for include_clinical in "" "--include_clinical"; do
                    echo "python model.py --data_types=$data --model_type=$model $sum_epts $calibrate $include_clinical"
                    output=$(python "$SCRIPT_DIR/model.py" --data_types="$data" --model_type="$model" $sum_epts $calibrate $include_clinical)
                    if [ $? -ne 0 ]; then
                        echo "Error executing python command. Exiting."
                        exit 1
                    fi
                    echo "Output: $output"
                    if [[ $data == "${data_types[0]}" && $model == "${model_types[0]}" && -z $sum_epts && -z $calibrate && -z $include_clinical ]]; then
                        header=$(echo "$output" | head -n 1)
                        echo "data,model,sum_epts,calibrate,include_clinical,$header" > "$OUTPUT_LOG"
                    fi
                    csv_line=$(echo "$output" | tail -n 1)
                    echo "$data,$model,$sum_epts,$calibrate,$include_clinical,$csv_line" >> "$OUTPUT_LOG"
                done
            done
        done
    done
done
