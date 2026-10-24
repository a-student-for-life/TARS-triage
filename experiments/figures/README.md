# Result figures

Scripts that draw the paper's result figures from the files committed under
`experiments/`. Run them from anywhere; output goes to `experiments/figures/out/`
(not committed).

| Script | Paper figure | Needs |
|---|---|---|
| `make_fig_examples.py` | Fig. 3, random browse crops per class | browse images and footprints (`browse_eval/fetch_data.py`, `gulf_eval/README.md`) |
| `make_fig_extra.py` | Fig. 5 CSIRO ROC and F1; Fig. 9 precision–recall; Fig. 11 Layer 3; Fig. 12 latency | committed files only (`revision/baseline_oof.py` and `revision/latency_waits.py` make two inputs) |
| `make_fig_roc.py` | Fig. 6, ROC on browse crops | committed files only |
| `make_fig_size.py` | Fig. 7, AUC by slick size | committed files only |
| `make_fig_crop_examples.py` | Fig. 8, example decisions | Mediterranean browse images |
| `make_fig_offcentre.py` | Fig. 10, off-centre slicks | committed files only |

Fig. 1 (system overview) is drawn in TikZ in the manuscript, Fig. 2 shows CSIRO chips and
Fig. 4 is `browse_eval/orientation_check_bbox.jpg`.
