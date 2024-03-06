from datetime import datetime
import os

CURRENT_TIME = datetime.now()
STRING_TIME = CURRENT_TIME.strftime("%Y-%m-%d-%H%M")
OUTPUT_DIR = os.path.join(f"./output/0_imported_circuits/{STRING_TIME}")

QASM_FILE_PATH= "/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/B-WGAN" \
                 "-Evol/input/final_best_ciruit_U_CNOT.qasm"


