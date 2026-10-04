# Compression Strategy, Not Compression Ratio

**Explanation-fidelity trade-offs in quantized and pruned cardiac risk models at matched footprint**

Code for the paper of the same name (under double-blind review).  It trains a small cardiac-risk DNN, compresses it into TensorFlow Lite variants, and measures how far each variant's SHAP attributions move from the float32 reference. Accuracy, size, and latency are reported alongside.

> **Scope.** The testbed is one retrospective tabular dataset (IEEE DataPort D1, n = 1,190). Nothing here validates a deployed wearable or sensor stream. Latency numbers come from a cloud-CPU proxy, not edge hardware.

---

## Main finding

Two variants with the **same 17.25 KB footprint** have very different explanation fidelity:

- **int8** leaves attributions almost unchanged (Spearman ρ = 0.992).
- **pruned+int8** (50% magnitude pruning with fine-tuning) moves them more than replacing the SHAP background sample does (ρ = 0.886, against a background-sensitivity reference of 0.939).

Accuracy does not predict this. Fine-tuning alone keeps accuracy (0.871) but lowers fidelity (IWEF 0.921). int8 loses slightly more accuracy (0.869) and keeps more fidelity (IWEF 0.978). The ablation ladder attributes the drift to **pruning with fine-tuning, not to quantization**.

## Results

Test set n = 238. DNN rows are mean ± std over 10 seeds (42–51). Each seed re-runs split → train → compress → explain with exact SHAP over the full test set.

| Model | Acc. | Size (KB) | Top-k Jaccard | Spearman ρ | IWEF |
|---|---|---|---|---|---|
| float32 (reference) | 0.871 ± 0.021 | 40.76 | 1 | 1 | 1 |
| float32 vs. itself, different background (**BSR**) | – | – | 0.781 | 0.939 | 0.829 |
| float16 | 0.871 ± 0.021 | 22.29 | 0.9994 ± 0.0007 | 0.9999 ± 0.0001 | 0.9996 ± 0.0003 |
| int8 | 0.869 ± 0.021 | 17.25 | 0.972 ± 0.015 | 0.992 ± 0.005 | 0.978 ± 0.010 |
| **pruned+int8** | 0.828 ± 0.015 | 17.25 | 0.701 ± 0.043 | 0.886 ± 0.027 | 0.800 ± 0.031 |
| *ablation:* fine-tune only | 0.871 ± 0.022 | 40.76 | 0.891 ± 0.043 | 0.973 ± 0.013 | 0.921 ± 0.034 |
| *ablation:* pruned (f32) | 0.831 ± 0.013 | 40.76 | 0.706 ± 0.040 | 0.887 ± 0.029 | 0.802 ± 0.030 |

Classical baselines (single run): SVM 0.878, Random Forest 0.933, KNN 0.899 accuracy. The DNN is the deployment target because it has a direct quantization and pruning path, not because it is the strongest classifier.

Other results:

- **McNemar's exact test** finds no significant prediction-level change for any variant. It has little power at 14 discordant pairs, so the consistent accuracy drop across seeds is the stronger evidence for pruning's predictive cost.
- **Permutation importance** (a second, gradient-free explainer) orders the variants the same way.
- **Robustness:** on a de-duplicated copy (n = 918, 5 seeds) the variant ordering by IWEF is unchanged, and absolute IWEF shifts by at most 0.019.
- **Deployment rule (Algorithm 1)** with τ_f = 0.90 and τ_a = 0.85 selects **int8**. pruned+int8 is never selected because it has int8's size and lower fidelity.

## How to read the fidelity scores

- **Exact SHAP** means all 2¹¹ = 2,048 feature coalitions are enumerated per sample. The background is still a sample (a 10-center k-means summary of 50 training rows), so the explainer is exact given that background, not background-free.
- **BSR (background-sensitivity reference).** The same float32 model is explained under two independent background draws. This is a yardstick for scale, **not a ceiling or a null distribution**. A variant scoring below it changes attributions by more than an arbitrary but defensible modelling choice does. A score above it does not mean the variant explains "better than the original."
- **IWEF (Importance-Weighted Explanation Fidelity)** weights rank displacement by the reference model's global feature importances. Each sample is normalized by its maximum attainable weighted displacement (Hungarian algorithm), so the score is tight on [0, 1].
- **"±"** is always the standard deviation across seeds. Confidence intervals, where used, are Student-t half-widths over 10 seeds. Seeds re-split the same 1,190 rows, so they are not independent samples of the population.
- Precision, recall, F1, and AUC are single-run (primary split, seed 42).

## Repository contents

| Path | Description |
|---|---|
| `IoMT_Cardiac_Risk_Edge_Compression_Pipeline_v4.ipynb` | End-to-end pipeline, from data loading to final figures. CPU only; runs in Colab or local Jupyter. |
| `outputs/` | Generated on run: CSVs, figures, LaTeX macros, TFLite models, reproducibility manifest. |

This is a research notebook, not a packaged library. Sections can be re-run independently, and each writes to `OUT_DIR` (`./outputs` locally, `/content/outputs` on Colab).

## Notebook map

| Section | Content |
|---|---|
| 0–3 | Setup, D1 loading, preprocessing, stratified splits |
| 4 | SVM / RF / KNN baselines; DNN 24-config grid search (4 × 3 × 2); final refit |
| 5 | TFLite compression: float32, float16, int8, pruned+int8, plus ablations (fine-tune only, pruned f32); sparsity sweep |
| 6 | McNemar's exact test |
| 7 | Size and latency (CPU proxy); confusion matrices and ROC curves |
| 8 | **Explanation-fidelity engine**: BSR, Jaccard / Spearman / IWEF, exact SHAP for all variants, DNN-SHAP vs. RF plausibility check, permutation importance, Shapley-linearity rank-swap bound |
| 9 | Multi-seed protocol (10 seeds) |
| 10 | De-duplication sensitivity (D1 has 272 exact duplicate rows) |
| 11 | Accuracy vs. size vs. fidelity trade-off plot |
| 12 | Algorithm 1 (uncertainty-aware tier selection); paired Wilcoxon tests |
| 13 | `table1_final.csv` and `paper_macros.tex` |
| 14–16 | Workflow diagram, seven-panel synthesis figure, system architecture |
| 17 | Raspberry Pi / ESP32 benchmark script (run **outside** Colab) |
| 18–19 | Reproducibility manifest; artifact summary |

> Some code comments and headings in the notebook refer to "noise floor". The paper's term is **background-sensitivity reference (BSR)**. They are the same quantity.

## Getting started

Requirements: Python 3, CPU only. Versions used for the reported results: Python 3.13, TensorFlow 2.20 (legacy Keras), SHAP 0.52. Exact library versions are recorded in `reproducibility_manifest.json` after a run.

```bash
pip install shap tensorflow-model-optimization
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy
```

Open the notebook in Colab or Jupyter and run top to bottom. **Start with `RUN_MODE = "smoke"`** (3 seeds). The full run (`RUN_MODE = "full"`, 10 seeds) takes about 20 minutes per seed for the Keras reference SHAP alone, and about 12–16 seconds per TFLite variant.

**Data:** the IEEE DataPort D1 heart disease dataset is public. Download it from IEEE DataPort and follow the loading cell in Section 1. The paper's run used a file with MD5 `89331abbd1786b4942ab8c588b832d39`. The dataset's own license terms apply and are not reproduced here.

## Reproducing the paper's numbers

1. Run Sections 0–8 for a single-seed check of every metric.
2. Run Section 9 with `N_SEEDS = 10` for the mean ± std table.
3. Run Section 10 for the de-duplication check.
4. Run Section 12 for Algorithm 1 and the paired Wilcoxon tests.
5. Run Section 13 to regenerate `paper_macros.tex`. Numbers quoted in the manuscript are written from the results tables, so paper and code cannot silently diverge.
6. For real edge timings, copy the Section 17 script to a Raspberry Pi or ESP32-class device.

TensorFlow op determinism is enabled, but exact numbers can still vary slightly across library versions and hardware.

## Exported artifacts

Written to `OUT_DIR`: baseline and DNN comparison tables, the 24-config grid log, McNemar results, size and latency benchmarks, the sparsity sweep, confusion matrices and ROC curves, single-run and multi-seed fidelity tables (including the BSR row), the rank-swap bound check, the permutation-importance cross-check, de-duplication results, paired Wilcoxon tests, `tier_selection.json`, `table1_final.csv`, `paper_macros.tex`, the trade-off and seven-panel figures, the pipeline diagram, `reproducibility_manifest.json`, all `.tflite` models, and `X_test.npy` / `y_test.npy` for hardware benchmarking.

## Limitations

- **One dataset.** D1 is a single retrospective tabular dataset with 272 duplicates. The fidelity ordering survives de-duplication (n = 918), but other cohorts and real sensor data are untested.
- **One pruning level.** Fidelity was measured only at 50% sparsity. The sparsity sweep reports accuracy and gzip size, not fidelity, so **no fidelity-versus-compression-ratio curve is claimed**. The matched-footprint comparison (int8 vs. pruned+int8 at 17.25 KB) is what supports the title. Raw `.tflite` size is unchanged by pruning because storage is dense. Only gzip size shrinks (10.95 → 5.22 KB from 30% to 90% sparsity).
- **One int8 calibration** and one SHAP background construction. The second explainer (permutation importance) is global only.
- **Weak plausibility check.** DNN-SHAP agrees only weakly with RF importances (ρ = 0.418, p = 0.20). Sex ranks 2nd under SHAP and 9th for RF.
- **Low-power tests.** McNemar's test has little power at n = 238. Pooled Wilcoxon p-values are descriptive because samples within a seed share one model.
- **Bound check is not a proof.** The rank-swap bound (Proposition 2) uses ε estimated on sample inputs, which lower-bounds the true supremum. The bound is informative only when compression barely perturbs outputs (float16). For int8 and the pruned variants it flags 90–100% of feature pairs as at risk.
- **Latency is a CPU proxy.** int8 is not faster on that CPU. Physical-hardware results are future work.
- **DNN trails Random Forest** in accuracy by design (see above).
- **Two explanation methods only:** SHAP plus permutation importance, and one compression family (magnitude pruning, post-training quantization).

## Related work note

Prior cardiac-XAI work we reviewed explains uncompressed models, and prior edge or federated cardiac work we reviewed reports no explanation analysis. The paper's related-work section and comparison table list the specific studies. This is a statement about that sample, not an exhaustive survey.

## Citation

Under review. Until the paper is public, please cite the repository. After acceptance, the BibTeX entry will be added here.

```bibtex
@unpublished{hira2027compression,
  title  = {Compression Strategy, Not Compression Ratio: Explanation-Fidelity
            Trade-offs in Quantized and Pruned Cardiac Risk Models at Matched Footprint},
  author = {Hira, Md Irfanul Kabir and Rahman, Anichur and Rana, Md Shohel},
  note   = {Under review, AISTATS 2027}
}
```

## License

The D1 dataset has its own terms via IEEE DataPort. A code license will be added after the review period.
