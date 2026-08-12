"""Canonical stems for paper figures (final_fig1–4, S_fig1–4).

Legacy FIG* stems remain only where helper CLIs still write intermediate panels.
"""

from __future__ import annotations

import os
from typing import List, Tuple

# Legacy panel stems still referenced by helper modules
FIG1 = 'fig1_basic_generation_quality'
FIG2 = 'fig2_frequency_domain'
FIG3 = 'fig3_visualization_experiments'
FIG4 = 'fig4_erp_topo_100ms'
FIG5 = 'fig5_erp_heatmap'
FIG6 = 'fig6_band_channel_similarity'
FIG7 = 'fig7_eeg_image_similarity'
FIG8 = 'fig8_retrieval_gallery'
FIG10 = 'fig10_eeg_tsne_umap_joint'
FIG11 = 'fig11_matched_pair_distance'
FIG12 = 'fig12_encoder_pc1'
FIG14 = 'fig14_rsa_rdm'
FIG15 = 'fig15_rsa_per_subject'
FIG16 = 'fig16_rsa_permutation_null'
FIG17 = 'fig17_generator_ablation'
FIG19 = 'fig19_loss_ablation'
FIG20 = 'fig20_loss_ablation_delta'

FINAL_FIG1 = 'final_fig1_EEG_Response_Prediction'
FINAL_FIG2 = 'final_fig2_Frequency_Spectral_Fidelity'
FINAL_FIG3 = 'final_fig3_EEG_Representational_Alignment'
FINAL_FIG4 = 'final_fig4_Generator_Loss_Ablation'

S_FIG1 = 'S_fig1_prediction_quality_supp'
S_FIG2 = 'S_fig2_single_image_waveforms'
S_FIG3 = 'S_fig3_representational_alignment_supp'
S_FIG4 = 'S_fig4_alljoined_selected_panels'

FIGURE_CATALOG: List[Tuple[str, str]] = [
    (FINAL_FIG1, 'EEG response prediction'),
    (FINAL_FIG2, 'Frequency / spectral fidelity'),
    (FINAL_FIG3, 'EEG representational alignment'),
    (FINAL_FIG4, 'Generator / loss ablation'),
    (S_FIG1, 'Prediction quality (supplement)'),
    (S_FIG2, 'Single-image waveforms (supplement)'),
    (S_FIG3, 'Representational alignment (supplement)'),
    (S_FIG4, 'Alljoined selected panels (supplement)'),
]

ALL_FIG_STEMS: Tuple[str, ...] = tuple(stem for stem, _ in FIGURE_CATALOG)


def fig_path(stem: str) -> str:
    """Absolute path prefix (no extension) under the active FIG_DIR."""
    from analysis.eeg_gen_eval import config as cfg
    return os.path.join(cfg.FIG_DIR, stem)


def fig_index(stem: str) -> int:
    """Return 1-based figure number from stem, or 0 if unknown."""
    for i, (s, _) in enumerate(FIGURE_CATALOG, start=1):
        if s == stem:
            return i
    return 0
