from datetime import datetime
import os
import csv

CURRENT_TIME = datetime.now()
STRING_TIME = CURRENT_TIME.strftime("%Y-%m-%d-%H%M")
OUTPUT_DIR = os.path.join(f"./output/0_imported_circuits/{STRING_TIME}")

INPUT_FOLDER = "/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/B-WGAN" \
               "-Evol/input/24_03_19_15_29_26/"

QASM_FILE_PATH= INPUT_FOLDER + "final_best_circuit.qasm"
METADATA_FILE_PATH = INPUT_FOLDER + "metadata.csv"

#### Image parameters ####
CLASSES = "01"
DATASET_STR = "mnist"
IMAGE_SIZE = 28

#### Training parameters ####
RANDN = False
BATCH_SIZE = 25
N_EPOCHS = 50
