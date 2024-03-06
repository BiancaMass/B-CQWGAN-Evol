import training_configs as config
from train_imported import train_imported

output_dir = config.OUTPUT_DIR
qasm_file_path = config.QASM_FILE_PATH

print(f'Calling circuit with the following params:\n'
      f'output_dir = {output_dir}\ninput_circuit = {qasm_file_path}')

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
