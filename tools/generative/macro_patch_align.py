"""
Alineación ECC + Flujo Óptico Farneback y extracción de Alta Frecuencia (High-Pass) para parches
macro Nano Banana ("posición humana") en Montaña Hero 360° v05.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def _ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (0, 0), sigma)
    mu_b = cv2.GaussianBlur(b, (0, 0), sigma)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a2 = np.maximum(cv2.GaussianBlur(a * a, (0, 0), sigma) - mu_a2, 0.0)
    sig_b2 = np.maximum(cv2.GaussianBlur(b * b, (0, 0), sigma) - mu_b2, 0.0)
    sig_ab = cv2.GaussianBlur(a * b, (0, 0), sigma) - mu_ab
    num = (2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    return num / np.maximum(den, 1e-8)


def align_and_extract_patch(
    ref_path: Path,
    nb_path: Path,
    mask_path: Path,
    out_hp_png16: Path,
    sigma_hp: float = 6.0,
    max_flow_px: float = 2.5,
    hp_clip: float = 0.18,
) -> Dict[str, Any]:
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    nb_bgr = cv2.imread(str(nb_path), cv2.IMREAD_COLOR)
    msk_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if ref_bgr is None or nb_bgr is None or msk_raw is None:
        return {"accepted": False, "reason": "missing_input"}

    H, W = ref_bgr.shape[:2]
    if nb_bgr.shape[:2] != (H, W):
        nb_bgr = cv2.resize(nb_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if msk_raw.shape[:2] != (H, W):
        msk_raw = cv2.resize(msk_raw, (W, H), interpolation=cv2.INTER_LINEAR)

    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    nb_rgb = cv2.cvtColor(nb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (msk_raw.astype(np.float32) / 255.0) > 0.50

    if int(mask.sum()) < 2000:
        return {"accepted": False, "reason": "mask_too_small"}

    ref_lum = 0.2126 * ref_rgb[..., 0] + 0.7152 * ref_rgb[..., 1] + 0.0722 * ref_rgb[..., 2]
    nb_lum = 0.2126 * nb_rgb[..., 0] + 0.7152 * nb_rgb[..., 1] + 0.0722 * nb_rgb[..., 2]

    # Alineación ECC sobre magnitud de gradiente Sobel normalizada (insensible al cambio de albedo en valles)
    ref_grad = np.hypot(cv2.Sobel(ref_lum, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ref_lum, cv2.CV_32F, 0, 1, ksize=3))
    nb_grad = np.hypot(cv2.Sobel(nb_lum, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(nb_lum, cv2.CV_32F, 0, 1, ksize=3))
    ref_grad_n = np.clip(ref_grad / max(float(np.percentile(ref_grad[mask], 95)), 1e-4), 0.0, 1.0)
    nb_grad_n = np.clip(nb_grad / max(float(np.percentile(nb_grad[mask], 95)), 1e-4), 0.0, 1.0)

    warp = np.eye(2, 3, dtype=np.float32)
    mask_u8 = (mask.astype(np.uint8) * 255)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-5)
        _, warp = cv2.findTransformECC(
            ref_grad_n,
            nb_grad_n,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=mask_u8,
            gaussFiltSize=5,
        )
    except cv2.error:
        warp = np.eye(2, 3, dtype=np.float32)

    if np.hypot(warp[0, 2], warp[1, 2]) > 8.0 or np.linalg.norm(warp[:, :2] - np.eye(2, dtype=np.float32)) > 0.035:
        warp = np.eye(2, 3, dtype=np.float32)

    nb_ecc = cv2.warpAffine(
        nb_rgb,
        warp,
        (W, H),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )
    nb_ecc_lum = 0.2126 * nb_ecc[..., 0] + 0.7152 * nb_ecc[..., 1] + 0.0722 * nb_ecc[..., 2]
    ref_l_m = float(ref_lum[mask].mean())
    nb_ecc_lum_norm = np.clip(nb_ecc_lum * (ref_l_m / max(float(nb_ecc_lum[mask].mean()), 1e-4)), 0.0, 1.0)

    # Flujo óptico denso Farneback ponderado por gradiente en crestas (evita deformar cojines de musgo en valles)
    flow = cv2.calcOpticalFlowFarneback(
        (ref_lum * 255.0).astype(np.uint8),
        (nb_ecc_lum_norm * 255.0).astype(np.uint8),
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=25,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    grad_w = np.clip(cv2.GaussianBlur(ref_grad, (0, 0), 8.0) / max(float(np.percentile(ref_grad[mask], 80)), 1e-4), 0.0, 1.0)
    flow_weighted = flow * grad_w[..., None]
    flow_clipped = cv2.GaussianBlur(np.clip(flow_weighted, -max_flow_px, max_flow_px), (0, 0), 5.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = xx + flow_clipped[..., 0]
    map_y = yy + flow_clipped[..., 1]
    nb_aligned = cv2.remap(
        nb_ecc,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )
    nb_aligned_lum = 0.2126 * nb_aligned[..., 0] + 0.7152 * nb_aligned[..., 1] + 0.0722 * nb_aligned[..., 2]

    # Puertas de calidad en paso bajo (sigma=6 px)
    ref_low_rgb = cv2.GaussianBlur(ref_rgb, (0, 0), sigma_hp)
    nb_low_rgb = cv2.GaussianBlur(nb_aligned, (0, 0), sigma_hp)
    ref_low_l = 0.2126 * ref_low_rgb[..., 0] + 0.7152 * ref_low_rgb[..., 1] + 0.0722 * ref_low_rgb[..., 2]
    nb_low_l = 0.2126 * nb_low_rgb[..., 0] + 0.7152 * nb_low_rgb[..., 1] + 0.0722 * nb_low_rgb[..., 2]

    # Igualar ganancia global paso bajo antes de medir SSIM y ΔE (el High-Pass descarta exposición global)
    gain_l = float(ref_low_l[mask].mean()) / max(float(nb_low_l[mask].mean()), 1e-4)
    nb_low_norm = np.clip(nb_low_rgb * gain_l, 0.0, 1.0)
    nb_low_l_norm = np.clip(nb_low_l * gain_l, 0.0, 1.0)

    ssim_m = _ssim_map(ref_low_l, nb_low_l_norm, sigma=4.0)
    ssim_low = float(ssim_m[mask].mean())

    flow_mag = np.hypot(flow_clipped[..., 0], flow_clipped[..., 1])
    mean_flow_px = float(flow_mag[mask].mean())

    # Ridge recall dentro de 5 px y detección de intrusión al cielo
    ref_sm = cv2.GaussianBlur(ref_lum, (0, 0), 2.0)
    nb_sm = cv2.GaussianBlur(nb_aligned_lum, (0, 0), 2.0)
    gr = np.hypot(cv2.Sobel(ref_sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ref_sm, cv2.CV_32F, 0, 1, ksize=3))
    gn = np.hypot(cv2.Sobel(nb_sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(nb_sm, cv2.CV_32F, 0, 1, ksize=3))
    mask_er = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if int(mask_er.sum()) > 500:
        edges_ref = (gr > np.percentile(gr[mask_er], 80)) & mask_er
        edges_nb = (gn > np.percentile(gn[mask_er], 75)).astype(np.uint8)
        dil_nb = cv2.dilate(edges_nb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        ridge_recall_5px = float(dil_nb[edges_ref].mean()) if np.any(edges_ref) else 1.0
    else:
        ridge_recall_5px = 1.0

    sky_er = cv2.erode((~mask).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if int(sky_er.sum()) > 500 and int(mask_er.sum()) > 500:
        edge_thr = max(float(np.percentile(gr[mask_er], 80)), 0.08)
        sky_intrusion_pct = float(((gn[sky_er] > edge_thr) & (gr[sky_er] < 0.03)).mean())
    else:
        sky_intrusion_pct = 0.0

    ref_lab = cv2.cvtColor(np.clip(ref_low_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    nb_lab = cv2.cvtColor(np.clip(nb_low_norm * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    # Escala CIE Lab OpenCV: L [0..255]->[0..100], a/b -128
    dL = (nb_lab[..., 0] - ref_lab[..., 0]) * (100.0 / 255.0)
    da = nb_lab[..., 1] - ref_lab[..., 1]
    db = nb_lab[..., 2] - ref_lab[..., 2]
    delta_e = float(np.sqrt(dL * dL + da * da + db * db)[mask].mean())
    chroma_delta_e = float(np.sqrt(da * da + db * db)[mask].mean())

    # Extracción High-Pass en log-espacio (luminancia plena + 0.35x croma suave, acotado ±0.18)
    log_nb = np.log(np.maximum(nb_aligned, 0.015))
    log_nb_blur = cv2.GaussianBlur(log_nb, (0, 0), sigma_hp)
    hp_rgb = log_nb - log_nb_blur

    hp_lum = 0.2126 * hp_rgb[..., 0] + 0.7152 * hp_rgb[..., 1] + 0.0722 * hp_rgb[..., 2]
    hp_chroma = hp_rgb - hp_lum[..., None]
    hp_out = np.clip(hp_lum[..., None] + 0.35 * hp_chroma, -hp_clip, hp_clip)
    if sky_intrusion_pct > 0.05:
        hp_mask = (cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0).astype(np.float32)
    else:
        hp_mask = mask.astype(np.float32)
    hp_out = hp_out * hp_mask[..., None]

    hp_std_val = float(hp_out[mask].std())
    sky_ok = (sky_intrusion_pct <= 0.05) or (ridge_recall_5px >= 0.82)
    accepted = (
        (ssim_low >= 0.58)
        and (ridge_recall_5px >= 0.72)
        and sky_ok
        and (chroma_delta_e <= 9.5)
        and (delta_e <= 21.0)
        and (hp_std_val >= 0.025)
    )

    # Codificar 16-bit PNG en rango [-0.20, +0.20] -> [0, 65535] con byte bajo en cero (~6x menor peso DEFLATE)
    out_hp_png16.parent.mkdir(parents=True, exist_ok=True)
    enc = np.clip(0.5 + hp_out / 0.40, 0.0, 1.0)
    s = np.clip(np.round((enc - 0.5) * 256.0), -128, 127).astype(np.int32)
    enc_u16 = ((s + 128) << 8).astype(np.uint16)
    cv2.imwrite(
        str(out_hp_png16),
        cv2.cvtColor(enc_u16, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 6],
    )

    return {
        "accepted": bool(accepted),
        "ssim_low": round(ssim_low, 4),
        "ridge_recall_5px": round(ridge_recall_5px, 4),
        "sky_intrusion_pct": round(sky_intrusion_pct, 4),
        "mean_flow_px": round(mean_flow_px, 4),
        "deltaE_low": round(delta_e, 4),
        "chroma_deltaE": round(chroma_delta_e, 4),
        "hp_std": round(hp_std_val, 5),
        "hp_file": str(out_hp_png16),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--macro-dir", required=True)
    p.add_argument("--sigma-hp", type=float, default=6.0)
    args = p.parse_args()

    mdir = Path(args.macro_dir)
    cams = json.loads((mdir / "macro_cameras.json").read_text(encoding="utf-8"))
    report: Dict[str, Any] = {"patches": {}, "accepted": 0, "rejected": 0}
    for k in sorted(cams.keys()):
        ref_p = Path(cams[k]["pbr_render"])
        msk_p = Path(cams[k]["mask"])
        nb_p = mdir / "nb" / f"patch_{k}_nb.png"
        if not nb_p.exists():
            nb_p = mdir / "nb" / f"patch_{k}_nb.jpg"
        hp_p = mdir / "aligned_hp" / f"patch_{k}_hp16.png"
        res = align_and_extract_patch(ref_p, nb_p, msk_p, hp_p, sigma_hp=args.sigma_hp)
        report["patches"][k] = res
        if res.get("accepted"):
            report["accepted"] += 1
        else:
            report["rejected"] += 1
        print(f"  [Macro {k}] {res}")

    (mdir / "macro_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[MacroAlign Done] accepted={report['accepted']}/{len(cams)}")


if __name__ == "__main__":
    main()
