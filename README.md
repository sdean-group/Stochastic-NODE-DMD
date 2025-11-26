# Sparse-to-Field Reconstruction via Stochastic Neural Dynamic Mode Decomposition

Official implementation of Stochastic NODE--DMD-- Probabilistic modeling of DMD supporting sparse observation and uncertainty quantification. 

by [Yujin Kim](https://yujin1007.github.io/) and [Sarah Dean &dagger;](https://sdean-group.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05294-df2a2a.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/pdf/2511.20612)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
[![Website](https://img.shields.io/badge/🔗-WebSite-black?style=for-the-badge)](https://sdean-group.github.io/Stochastic-NODE-DMD/)
<!-- [![Summary](https://img.shields.io/badge/-Summary-1DA1F2?logo=x&logoColor=white&labelColor=gray&style=for-the-badge)](https://x.com/wzhao_nlp/status/1896962009918525730) -->

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
  <img width="1000" src="assets/reconstruction3.png">
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
After evaluation, image files (including gif) and performance summary note will be saved in `<SAVE_DIR>/autoreg_reconstruction` and `<SAVE_DIR>/teacher_forcing_reconstruction`.


## 📜 Citation

If you build on our work or find it useful, please cite it using the following bibtex.

```bibtex
@misc{kim2025sparsetofieldreconstructionstochasticneural,
      title={Sparse-to-Field Reconstruction via Stochastic Neural Dynamic Mode Decomposition}, 
      author={Yujin Kim and Sarah Dean},
      year={2025},
      eprint={2511.20612},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.20612}, 
}
```