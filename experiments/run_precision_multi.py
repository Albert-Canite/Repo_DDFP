import itertools
import numpy as np
import matplotlib.pyplot as plt

import core.config as C
from core.utils import load_images_rsna, snr_db
from ddfp.fp32_forward import run_network_fp32
from ddfp.baseline_forward import run_network_baseline
from ddfp.ddfp_forward import calibrate_ddfp, run_network_ddfp


def run_single_config(L, IN, W, ADC):
    # 配置初始化
    C.setup_config(L, IN, W, ADC)

    # 加载校准和测试图像
    cal_imgs = load_images_rsna(C.NUM_CALIBRATION)
    test_imgs = load_images_rsna(C.NUM_TEST)

    # 校准（得到 δ, α, β…）
    calibrate_ddfp(cal_imgs)

    # ----------------------------
    # 图像布局（与原始 Precision_multi 一致）
    # ----------------------------
    rows = C.NUM_TEST
    cols = 5
    titles = [
        "Original",
        "FP32 Ref",
        "DDFP Output",
        "Baseline Output",
        "DDFP - Baseline"
    ]
    cmaps = ['gray', 'viridis', 'viridis', 'viridis', 'seismic']

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, img in enumerate(test_imgs):

        # ----------------------------
        # 原图
        # ----------------------------
        if C.INPUT_SIGNED:
            orig_vis = img[0, 0] + 0.5
        else:
            orig_vis = img[0, 0]

        # ----------------------------
        # FP32 参考结果
        # ----------------------------
        ref = run_network_fp32(img, C.kernels)
        ref2d = ref[0, 0]

        # ----------------------------
        # Baseline 结果
        # ----------------------------
        base, _, _ = run_network_baseline(img)
        base2d = base[0, 0]

        # ----------------------------
        # DDFP 结果（未校准）
        # ----------------------------
        ddfp, _, _, _ = run_network_ddfp(img)
        ddfp_raw = ddfp[0, 0]

        # ==============================================================
        # ① 全局输出校准（论文中的 1×1 校准层）
        # ==============================================================

        ref_flat = ref2d.flatten()
        ddfp_flat = ddfp_raw.flatten()
        mask = np.abs(ref_flat) > 1e-6

        if np.sum(mask) > 10:
            a = np.dot(ref_flat[mask], ddfp_flat[mask]) / (
                np.dot(ddfp_flat[mask], ddfp_flat[mask]) + 1e-12
            )
        else:
            a = 1.0

        ddfp2d = a * ddfp_raw

        # ==============================================================
        # ② FP32 / Baseline / DDFP 使用相同 vmin/vmax
        # ==============================================================

        absmax = max(
            np.max(np.abs(ref2d)),
            np.max(np.abs(base2d)),
            np.max(np.abs(ddfp2d))
        )
        if absmax < 1e-12:
            absmax = 1e-6

        # 如果你想强制全部非负可改为 vmin=0
        # 这里遵循 FP32 范围（可能正负）
        vmin = -absmax
        vmax = absmax

        # ==============================================================
        # ③ 差分图（原版 ACIM 绘法）
        # ==============================================================

        diff = ddfp2d - base2d
        abs_p = float(np.percentile(np.abs(diff), 99.0))
        vlim_diff = abs_p if abs_p > 1e-8 else np.max(np.abs(diff)) + 1e-6
        thr = float(np.percentile(np.abs(diff), 98.0))

        # ==============================================================
        # ④ 计算 SNR
        # ==============================================================

        snr_d = snr_db(ref, ddfp)
        snr_b = snr_db(ref, base)
        delta_snr = snr_d - snr_b

        imgs = [orig_vis, ref2d, ddfp2d, base2d, diff]

        # ----------------------------
        # 绘五列子图
        # ----------------------------
        for j in range(cols):
            ax = axes[idx, j]
            ax.set_title(titles[j], fontsize=12)

            if j == 1:  # FP32
                im = ax.imshow(imgs[j], cmap=cmaps[j], vmin=vmin, vmax=vmax)
            elif j == 2:  # DDFP (校准后)
                im = ax.imshow(imgs[j], cmap=cmaps[j], vmin=vmin, vmax=vmax)
            elif j == 3:  # Baseline
                im = ax.imshow(imgs[j], cmap=cmaps[j], vmin=vmin, vmax=vmax)
            elif j == 4:  # diff
                im = ax.imshow(imgs[j], cmap=cmaps[j],
                               vmin=-vlim_diff, vmax=vlim_diff)
                ax.contour(imgs[j], levels=[thr], colors=["yellow"], linewidths=0.7)
            else:  # 原图
                im = ax.imshow(imgs[j], cmap=cmaps[j])

            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

            # 添加 SNR 文本（仅 DDFP / Baseline）
            if j == 2:
                ax.text(
                    0.02, 0.05,
                    f"SNR={snr_d:.2f} dB\nΔ={delta_snr:.2f}",
                    color="white", fontsize=9,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none")
                )
            if j == 3:
                ax.text(
                    0.02, 0.05,
                    f"SNR={snr_b:.2f} dB",
                    color="white", fontsize=9,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none")
                )

            if j == 4:
                ax.text(
                    0.02, 0.02,
                    "Blue = DDFP better\nRed = Baseline better",
                    color="white", fontsize=8,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none")
                )

    plt.tight_layout()

    out_name = f"DDFP_L{L}_IN{IN}_W{W}_ADC{ADC}.png"
    out_path = C.OUTPUT_DIR / out_name
    fig.savefig(out_path, dpi=150)

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    configs = list(itertools.product(
        C.NUM_LAYERS_LIST,
        C.INPUT_BITS_LIST,
        C.WEIGHT_BITS_LIST,
        C.ADC_BITS_LIST
    ))

    for (L, IN, W, ADC) in configs:
        print(f"\n=== Running Config: L={L} IN={IN} W={W} ADC={ADC} ===")
        run_single_config(L, IN, W, ADC)
