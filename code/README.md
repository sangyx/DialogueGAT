# DialogueGAT

* Dependencies: PyTorch, DGL
* Download the data from <https://drive.google.com/file/d/1fhTQdoWxQPfL33AUaCEbgk6GGBV-LvU1/view?usp=sharing>, place it in `data`.
* Running all experiments: 
  ```
  sh run.sh
  ```
* Runing a single task:
  ```
  python -W ignore -u train.py --use_gpu --v_past --year ${year} --target ${target}
  ```