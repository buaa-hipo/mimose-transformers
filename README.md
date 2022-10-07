## Requirements

- V100
- Docker with functional NVIDIA GPU support

## Install

1. Create a docker container with NVIDIA GPU enabled

   ```bash
   docker run --name mimose -itd --gpus all -v <dataset_path>:/opt/dataset pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel bash
   docker exec -it mimose bash
   ```

2. Install Git using `apt`

   ```bash
   chmod 777 /tmp # apt update would fail without this
   apt update
   apt install -y git
   ```

3. Setup conda, create a new env and install PyTorch

   ```bash
   # Setup conda
   conda init
   . ~/.bashrc
   
   # Create conda env and install PyTorch
   conda create -n mimose python=3.9
   conda activate mimose
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   
   ```

4. Install `mimose-transformers` and necessary dependencies

   ```bash
   # Setup mimose-transformers repo and install dependencies
   git clone https://github.com/mimose-project/mimose-transformers && cd mimose-transformers
   pip install -v -e .
   pip install -r examples/pytorch/translation/requirements.txt
   pip install -r examples/pytorch/question-answering/requirements.txt
   pip install -r examples/pytorch/multiple-choice/requirements.txt
   pip install -r examples/pytorch/text-classification/requirements.txt
   ```

## Getting Started

1. Run the evaluation scripts for mimose:

   ```bash
   cd mimose-transformers
   # Run the evaluation all-in-one script!
   bash exp.sh
   ```

2. Check logs in `examples/pytorch/<task>/train_log` directory, where `<task>` could be one of `translation`, `question-answering`, `multiple-choice` or `text-classification`.

3. You can also run seperate evaluation scripts executed in `exp.sh` manually.
