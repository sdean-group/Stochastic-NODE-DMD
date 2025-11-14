import matplotlib.pyplot as plt
import os
import imageio

def plot_loss(avg_loss_list, out_dir, filename="loss.png"):
    epochs = list(range(len(avg_loss_list)))  # 0부터 시작하는 epoch index
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_reconstruction(coords, t, y_true, u_pred, mse, out_dir, vmin, vmax):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), c=y_true[:, 0].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(f'True Real Part at t={t}')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), c=u_pred[:, 0].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(f'Pred Real Part at t={t}, MSE: {mse:.3f}')
    plt.colorbar()
    save_name = os.path.join(out_dir, f'prediction_t{t}.png')
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    return imageio.imread(save_name)