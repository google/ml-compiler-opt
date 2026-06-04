# LLVM MLGO Inliner Demo

## 1. Build the Docker Image

From the repository root, build the demo environment container:

```bash
docker build -t ml-compiler-opt-llvm docs/llvm/
```

## 2. Run the Container

Run the container with volume mounts to persist corpus data, training logs, and checkpoints locally. This enables running TensorBoard on the host to monitor training.

```bash
docker run -it \
  -v "$(pwd):/work/ml-compiler-opt" \
  -v "$(pwd)/local_logs:/work/corpus/" \
  ml-compiler-opt-llvm /bin/bash
```

## 3. Run the Training Pipeline

Inside the container, execute the entire end-to-end pipeline:

```bash
./docs/llvm/run_everything.sh
```

---

## 4. Monitor Training Progress

To monitor the training progress using TensorBoard, run the following command on your **host machine** pointing to the mounted logs directory:

```bash
tensorboard --logdir local_logs
```

---

## 5. Script Flags Documentation

If you want to understand or customize the flags used in the training pipeline scripts (like `train_bc.py`), you can:
* **Run with `--help`**: Use the `--help` flag with any of the Python scripts to print all available command-line flags and their descriptions:
  ```bash
  PYTHONPATH=. python3 compiler_opt/rl/train_bc.py --help
  ```
* **View Source Code Definitions**: The flags are defined inline within their respective Python files (such as [train_bc.py](file:///usr/local/google/home/bmandalapu/dev/ml-compiler-opt/compiler_opt/rl/train_bc.py)).

