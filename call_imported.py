import csv

import training_configs as config
from train_imported import train_imported

input_dir = config.INPUT_FOLDER
output_dir = config.OUTPUT_DIR
qasm_file_path = config.QASM_FILE_PATH
metadata_file_path = config.METADATA_FILE_PATH

print(f'Calling circuit with the following params:\n'
      f'output_dir = {output_dir}\ninput_circuit = {qasm_file_path}')

if __name__ == "__main__":
    train_imported(classes_str=config.CLASSES,
                   dataset_str=config.DATASET_STR,
                   out_folder=output_dir,
                   randn=config.RANDN,
                   qasm_file_path=qasm_file_path,
                   batch_size=config.BATCH_SIZE,
                   n_epochs=config.N_EPOCHS,
                   image_size=config.IMAGE_SIZE,
                   metadata_file_path = config.METADATA_FILE_PATH
                   )
