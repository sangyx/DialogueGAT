# DialogueGAT

## Data
Refer to [preprocess.ipynb](data_sample\preprocess.ipynb) to construct the experiment dataset.

## Run the experiment

* Dependencies: PyTorch, DGL
* Running all experiments: 
  ```
  sh run.sh
  ```
* Runing a single task:
  ```
  python -W ignore -u train.py --use_gpu --v_past --year ${year} --target ${target}
  ```