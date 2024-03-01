import os
import math
from datetime import datetime

from train_imported import train_imported

current_time = datetime.now()
string_time = current_time.strftime("%Y-%m-%d-%H%M")
output_dir = os.path.join(f"./output/0_imported_circuits/{string_time}")

qasm_file_path= "/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/B-WGAN" \
                "-Evol/input/final_best_ciruit_maximize_3.qasm"

if __name__ == "__main__":
    train_imported(classes_str="01",
                   dataset_str="mnist",
                   out_folder=output_dir,
                   randn=False,
                   qasm_file_path=qasm_file_path,
                   batch_size=25,
                   n_epochs=50,
                   image_size=28
                   )

