# Sparse-to-Field Reconstruction via Stochastic Neural Dynamic Mode Decomposition

Official implementation of Stochastic NODE--DMD-- Probabilistic modeling of DMD supporting sparse observation and uncertainty quantification. 
by [Yujin Kim](https://yujin1007.github.io/) and [Sarah Dean &dagger;](https://sdean-group.github.io/)

**Stochastic NODE–DMD** is a probabilistic and interpretable framework for system identification.  

1. **Generative interpretation of DMD**  
   Reformulates classical Dynamic Mode Decomposition as a generative model.

2. **Neural implicit spatial representation**  
   Enables grid-free, continuous spatial reconstruction from sparse observations.

3. **Stochastic Neural ODE dynamics**  
   Augments the linear DMD drift with a stochastic latent ODE to capture nonlinear residuals  
   and propagate uncertainty.

Together, these components preserve the spectral interpretability of DMD while extending its applicability to sparse, noisy, and strongly nonlinear systems.
<p align="center">
  <img width="1000" src="assets/reconstruction3">
</p>

## 🛠 Environment Setup

We recommend using **conda**:

```bash
conda create -n node_dmd python=3.10
conda activate node_dmd
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### ⚡ GPU Support (PyTorch)

By default `requirements.txt` installs the CPU version of PyTorch.  
If you want GPU acceleration, install PyTorch matching your CUDA version.

For CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

More info: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).



## 🚀 Training

Go inside the `src/` directory.  
Use the `--dataset` flag to specify the training task.  
You may change the model, checkpoint, and loss output directories in `config/config.py` by modifying the `save_dir` argument.

```bash
python train.py --dataset <TASK>
```

### Available `<TASK>` Options

| Task Name     | Description |
|---------------|-------------|
| `synthetic`   | Synthetic sequence with four ground-truth modes |
| `gray_scott`  | Gray–Scott reaction–diffusion benchmark |
| `vorticity`   | 2D spectral vorticity flow dataset |
| `cylinder`    | Cylinder wake flow dataset |

> `vorticity` and `cylinder` datasets are generated with [torch-cfd](https://github.com/scaomath/torch-cfd)

---

## 📈 Evaluation (autoregressive rollout & teacher-forcing rollout)

Go inside the `src/` directory.  
Trained models are located at `./results/<TASK>/run1/`.

```bash
python eval.py --config_dir <SAVE_DIR> --dataset <TASK>
```

Replace `<SAVE_DIR>` with the directory containing your saved model.



## 📜 Citation

If you use this code or build upon it, please cite appropriately (to be updated after publication).
