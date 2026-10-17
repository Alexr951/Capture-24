# Recognising daily activities from a wrist sensor

A frozen self-supervised encoder with about 10 minutes of temporal context reaches κ 0.79 on held-out people, 0.10 above a random forest + HMM baseline and 0.04 above an ActiNet-style frozen encoder + HMM.

Human activity recognition on [CAPTURE-24](https://doi.org/10.5287/bodleian:NGx0JOMP5): 151 people who wore a wrist accelerometer for a day, labelled with 10 activities (sleep, sitting, standing, household chores, manual work, walking, mixed activity, vehicle, sports, bicycling) from camera photos taken every 20 to 40 seconds. The task is to label every 10 s window. The last 51 participants are held out, as in the dataset paper ([Chan et al., 2024](https://doi.org/10.1038/s41597-024-03960-3)).

I first reproduced the paper's random forest + HMM, then built a model that uses a self-supervised encoder and reads about 10 minutes of context at a time. It reaches κ 0.790 on the held-out participants. The dataset paper's best model reaches 0.735.

The write-up is [`paper.pdf`](paper.pdf), *Frozen self-supervised features and temporal context for free-living activity recognition on CAPTURE-24*.

![Camera labels, a random forest and the forest smoothed by an HMM over six hours of one participant](outputs/bands.png)

## Results

Cross-validation (CV) is 5-fold and grouped by participant on P001-P100. Held-out scores are on P101-P151, with 95% bootstrap intervals over participants. Macro-F1 averages F1 over the 10 activities, so rare ones count as much as sleep; κ is agreement beyond chance. Each new model's settings and selection rule were written down before any of its scores were seen.

| Model | Macro-F1, CV | Macro-F1, held out | κ, CV | κ, held out |
|---|---|---|---|---|
| Paper's CNN + HMM, as published (trained from scratch, not paired) | | 0.576 | | 0.735 |
| Random forest + HMM | 0.522 | 0.566 [0.524, 0.595] | 0.708 | 0.690 [0.653, 0.725] |
| Fine-tuned encoder + HMM | 0.578 | 0.619 [0.557, 0.659] | 0.741 | 0.742 [0.702, 0.778] |
| Frozen encoder + HMM | 0.603 | 0.638 [0.571, 0.681] | 0.776 | 0.750 [0.717, 0.783] |
| **Frozen encoder + BiGRU** | 0.580 | 0.642 [0.577, 0.677] | 0.814 | **0.790 [0.758, 0.819]** |
| Frozen encoder + BiGRU minus random forest + HMM, paired | +0.058 | +0.076 [0.021, 0.111] | +0.105 | +0.100 [0.076, 0.124] |
| Frozen encoder + BiGRU minus fine-tuned encoder + HMM, paired | | +0.022 [-0.025, 0.057] | | +0.048 [0.026, 0.071] |
| Frozen encoder + BiGRU minus frozen encoder + HMM, paired | | +0.004 [-0.025, 0.026] | | +0.040 [0.023, 0.057] |

![Held-out scores](outputs/results_new_model.png)

- The encoder is [`ssl-wearables`](https://github.com/OxWearables/ssl-wearables) harnet10 (Yuan et al., 2024), pretrained by the dataset's authors without labels on 700,000 person-days of UK Biobank data. Their tool ActiNet ([Acquah et al., 2025](https://arxiv.org/abs/2510.01712)) uses the same encoder with a trained final layer, followed by an HMM and a sleep correction. The first version of its paper keeps the encoder frozen, and the current version fine-tunes it. It predicts 4 intensity classes from 30 s windows, so its numbers are not comparable to the 10 classes here. On held-out participants, the frozen encoder + BiGRU is above the ActiNet-style frozen encoder + HMM by +0.040 κ [0.023, 0.057]. The macro-F1 difference is +0.004 [-0.025, 0.026]. The paper's CNN was trained from scratch, so part of the gain over it is pretraining.
- In cross-validation the encoder adds 0.070 κ without context and 0.034 with it. Context adds 0.056 on hand-made features and 0.020 on the encoder. Each helps less when the other is present, and with context the HMM no longer helps.
- Part of the gap to the random forest + HMM is the decision rule. The frozen encoder + BiGRU keeps the class frequencies in its output and the HMM models divide them out. With every model at its best rule, the cross-validation gain is 0.064 κ instead of 0.105.

## Notebooks

- [`01_reproduction_and_baseline.ipynb`](01_reproduction_and_baseline.ipynb) reproduces the paper's random forest + HMM (κ 0.660 against 0.637 published), builds the random forest + HMM baseline and measures two evaluation pitfalls.
- [`02_pretrained_encoder_and_context.ipynb`](02_pretrained_encoder_and_context.ipynb) finds where the baseline's errors are, adds the pretrained encoder and the BiGRU, scores them on the held-out participants and takes the gain apart.

## Limitations

- The held-out participants have been scored six times over the project: the random forest + HMM twice, the rerun of the paper's pipeline, the fine-tuned encoder + HMM, the frozen encoder + BiGRU and the frozen encoder + HMM. Everything in between was chosen on cross-validation.
- The encoder's authors are the dataset's authors and evaluated it on CAPTURE-24 for 4 classes. So it was never trained on these labels, but its design was not chosen independently of this benchmark.
- Held-out macro-F1 leans on activities few participants did (8 did sports), so the cross-validation column is safer to quote.
- Labelled windows almost never sit on a change of activity, so every score describes the inside of bouts.
- One fine-tuning recipe was tried, one seed was scored on the held-out set, and all the data is from one UK cohort wearing one device.
- The frozen encoder + BiGRU reads whole recordings in both directions, so it is offline only.

## Reproduce

Tested with Python 3.13, 16 GB of memory, 8 cores and an RTX 2070 SUPER.

1. Download `capture24.zip` (6.9 GB) from <https://doi.org/10.5287/bodleian:NGx0JOMP5> into `data/`.
2. `pip install -r requirements.txt` and `python -m ipykernel install --user --name capture24`. Versions are pinned because the paper's pipeline depends on how imbalanced-learn 0.13.0 and scikit-learn 1.6.1 compute out-of-bag output.
3. Part 1 (CPU): `jupyter nbconvert --execute --to notebook --inplace 01_reproduction_and_baseline.ipynb`. About 7 min with caches; the first run adds about 27 min.
4. Part 2 (GPU): the same command on `02_pretrained_encoder_and_context.ipynb`. Fits are cached in `data/fits/`; with the cache it takes about 10 min, and from scratch it trains 16 fine-tuned encoders (about 9 min each) and 62 recurrent networks (3-5 min each).
5. Tests: `python -m pytest` and `ruff check .` from the repository root.

The code is in the `capture24` package (`from capture24 import har`): `features.py` (windows and features), `har.py` (folds, forest, HMM, metrics, bootstraps) and `deep.py` (encoder and networks). Every score is saved in `outputs/results.json` and `outputs/results_new_model.json`.

## Licence

The code is MIT. The paper, `paper.pdf`, is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CAPTURE-24 is CC BY 4.0 (Chan et al., 2024) and is not included. The `ssl-wearables` encoder is under the University of Oxford's Academic Use Licence (non-commercial, no redistribution); it is not included and is downloaded from its authors at a pinned commit.
