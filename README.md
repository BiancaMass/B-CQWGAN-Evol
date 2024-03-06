# B-CQWGAN-Evol


## Running on DSRI GPU

### Setting up the virtual environment
To run on DSRI GPU, I had to change the pytorch and torchvision versions.
While I found this solution, I was running on `GPU 0: NVIDIA H100 PCIe (UUID: GPU-023a2d83-c6a4-abe9-f546-d5957c1427e2)`

1. Make a venv:
`python3.10 -m venv venv1`
2. Activate it:
`source venv1/bin/activate`

3. In the terminal:
- `pip install -r requirements.txt`
- `pip install qiskit`
- `pip install pennylane`

4. Then, to change pytorch and torchvision to te correct versions to be compatible with the DSRI 
   GPU:
 - `pip3 uninstall torch`

- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

- `pip uninstall torch torchvision`

- `pip install torchvision==0.17.1`


This should solve this error:

`NVIDIA H100 PCIe with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.`


### Setting up the workspace
Note: the main folder in remote is called `B-CQWGAN-EVOL-1/`
1. Under the main folder, create a folder named `input/`
2. In the `input\` folder, add whatever circuit file (`.qasm` file) you want to use for the PQWGAN 
algorithm. The file should be called something like `final_best_ciruit.qasm`.
3. Under the main folder, create a folder named `output/` to store the output from the 
   evolutionary algorithm.