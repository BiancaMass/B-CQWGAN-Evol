# B-CQWGAN-Evol

### Running on DSRI GPU
To run on DSRI GPU, I had to change the pytorch and torchvision versions.
While I found this solution, I was running on `GPU 0: NVIDIA H100 PCIe (UUID: GPU-023a2d83-c6a4-abe9-f546-d5957c1427e2)`
1. Add an `\input` folder and an input file (`.qasm`).

2. In the terminal:
- `pip install -r requirements.txt`
- `pip install qiskit`

3. Then, to change pytorch and torchvision to te correct versions to be compatible with the DSRI 
   GPU:
 - `pip3 uninstall torch`

- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

- `pip uninstall torch torchvision`

- `pip install torchvision==0.17.1`


This should solve this error:

`NVIDIA H100 PCIe with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.`