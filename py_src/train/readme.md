# Training

# 1 GPU train

Run:
```bash
source ~/.virtualenvs/snap_cluster_setup/bin/activate
cd ~/snap-cluster-setup
python py_src/train/simple_train2.py
```

# 1-8 GPUs 1 node train

Run:
```bash
source ~/.virtualenvs/snap_cluster_setup/bin/activate
cd ~/snap-cluster-setup
# accelerate launch py_src/train/simple_train2.py
```