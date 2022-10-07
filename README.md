# DialogueGAT

## ToDo
Provide detailed instruction to construct the experiment dataset.

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