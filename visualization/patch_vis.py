import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def show_patch_orthogonal(patch, title=None):
    """
    Show axial / coronal / sagittal slices
    patch shape: (D, H, W)
    """
    assert patch.ndim == 3, f"Expected 3D patch, got {patch.shape}"

    D, H, W = patch.shape
    z, y, x = D // 2, H // 2, W // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(patch[z], cmap="gray")
    axs[0].set_title(f"Axial (z={z})")

    axs[1].imshow(patch[:, y, :], cmap="gray")
    axs[1].set_title(f"Coronal (y={y})")

    axs[2].imshow(patch[:, :, x], cmap="gray")
    axs[2].set_title(f"Sagittal (x={x})")

    for ax in axs:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def scroll_z_axis(patch):
    """
    Scroll through Z axis using keyboard:
    ← / → or A / D
    """
    D = patch.shape[0]
    idx = D // 2

    fig, ax = plt.subplots()
    img = ax.imshow(patch[idx], cmap="gray")
    ax.set_title(f"Z slice {idx}/{D-1}")
    ax.axis("off")

    def on_key(event):
        nonlocal idx
        if event.key in ["right", "d"]:
            idx = min(idx + 1, D - 1)
        elif event.key in ["left", "a"]:
            idx = max(idx - 1, 0)
        else:
            return

        img.set_data(patch[idx])
        ax.set_title(f"Z slice {idx}/{D-1}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D lung nodule patch (.npy)"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .npy patch"
    )
    parser.add_argument(
        "--no_scroll",
        action="store_true",
        help="Disable Z-axis scrolling"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"File not found: {args.path}")

    patch = np.load(args.path)

    print(f"[INFO] Loaded: {args.path}")
    print(f"[INFO] Shape: {patch.shape}")
    print(f"[INFO] Value range: {patch.min():.3f} ~ {patch.max():.3f}")

    title = os.path.basename(args.path)

    # 正交视图
    show_patch_orthogonal(patch, title=title)

    # # Z 轴滚动
    # if not args.no_scroll:
    #     print("[INFO] Z-axis scrolling: use ← / → or A / D")
    #     scroll_z_axis(patch)


if __name__ == "__main__":
    main()
