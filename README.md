# Recognising daily activities from a wrist sensor

Human activity recognition on [CAPTURE-24](https://doi.org/10.5287/bodleian:NGx0JOMP5): 151 people who wore a wrist accelerometer for a day, labelled with 10 activities (sleep, sitting, standing, household chores, manual work, walking, mixed activity, vehicle, sports, bicycling) from camera photos taken every 20 to 40 seconds. The task is to label every 10 s window. The last 51 participants are held out, as in the dataset paper ([Chan et al., 2024](https://doi.org/10.1038/s41597-024-03960-3)).

I first reproduced the paper's random forest + HMM baseline, then built a model that uses a self-supervised encoder and reads about 10 minutes of context at a time. It reaches κ 0.790 on the held-out participants, against 0.735 for the paper's best model.

![Camera labels, a random forest (A) and the forest smoothed by an HMM (B) over six hours of one participant](outputs/bands.png)

## Results

Held-out participants, with 95% bootstrap intervals over participants. Macro-F1 averages F1 over the 10 activities, so rare ones count as much as sleep; κ is agreement beyond chance.

| Model | Macro-F1 | κ |
|---|---|---|
| Paper's random forest + HMM, rerun here | 0.535 [0.490, 0.566] | 0.660 [0.630, 0.692] |
| B: my random forest + HMM | 0.566 [0.524, 0.595] | 0.690 [0.653, 0.725] |
| Deep + HMM: encoder fine-tuned, + HMM | 0.619 [0.557, 0.659] | 0.742 [0.702, 0.778] |
| **Context: encoder frozen, + bidirectional GRU** | **0.642 [0.577, 0.677]** | **0.790 [0.758, 0.819]** |
| Context minus B, paired by participant | +0.076 [0.021, 0.111] | +0.100 [0.076, 0.124] |
| Paper's CNN + HMM, as published (trained from scratch) | 0.576 | 0.735 |

![Held-out scores](outputs/results_new_model.png)

- The encoder is [`ssl-wearables`](https://github.com/OxWearables/ssl-wearables) harnet10 (Yuan et al., 2024), pretrained by the dataset's authors without labels on 700,000 person-days of UK Biobank data. It never saw CAPTURE-24. The paper's CNN was trained from scratch, so part of the gain over it is pretraining.
- An ablation in cross-validation separates the two parts: the encoder adds 0.034 to 0.070 κ over hand-made features, and context adds 0.020 to 0.056. Context also makes the HMM unnecessary.
- Part of the gap to B is the decision rule. Context keeps the class frequencies in its output and the HMM models divide them out; with every model at its best rule, Context's cross-validation gain over B is 0.064 κ instead of 0.105.
- Macro-F1 gains are smaller and less certain. Manual work and mixed activity stay at F1 0.3 or below in cross-validation.

## Notebooks

[`capture24_har.ipynb`](capture24_har.ipynb), part 1:

- Window counts match the paper exactly (922,199), and the paper's pipeline, rerun from its code, scores κ 0.660 against 0.637 published.
- B adds about 0.03 κ over that rerun on the held-out participants (paired interval 0.001 to 0.058).
- Two pitfalls, measured. Splitting sequences where annotation stops leaks 0.02 macro-F1. And with too few trees, imblearn's balanced forest gives 29% of sleep and sitting windows no out-of-bag output at all; they become rows of zeros, which costs an HMM fitted on that output 0.075 κ.

[`capture24_new_model.ipynb`](capture24_new_model.ipynb), part 2:

- Two thirds of B's errors are more than a minute from any label change; label timing explains at most about 0.03 κ.
- Annotators leave an unlabelled gap at almost every change of activity (46 of 609,580 adjacent labelled pairs differ), so the HMM learns "almost never switch".
- The pretrained encoder, fine-tuned and frozen, the context model, held-out results, and the ablation above.

Cross-validation is grouped by participant. Each new model's settings and selection rule were written down before any of its scores were seen, and the held-out set was scored once per model. Some analyses cut from part 2 (per-participant κ, HMM floor and switching-rate sweeps, fine-tuning variants, a drift check) are in commit `2e6492d`.

## Limitations

- The held-out participants have been scored five times over the project (B twice, the rerun, Deep + HMM, Context). Everything in between was chosen on cross-validation.
- Held-out macro-F1 leans on activities few participants did (8 did sports). The cross-validation figures, 0.52 for B and 0.58 for Context, are safer to quote.
- Labelled windows almost never sit on a change of activity, so every score describes the inside of bouts.
- One fine-tuning recipe was tried, one seed was scored on the held-out set, and all the data is from one UK cohort wearing one device.
- The context model reads whole recordings in both directions, so it is offline only.

## Try it

```
python scripts/predict.py recording.csv.gz --out results/
```

Writes a timeline CSV and a plot of the day from a raw recording (100 Hz, in g, columns `time, x, y, z`) using the Context model in `models/context_v1.pt`. `--rule balanced` trades some agreement for more of the rare activities. It has only been validated on CAPTURE-24's device (Axivity AX3, wrist, ±8 g). The model file is in Git LFS, so clone with Git LFS installed or run `git lfs pull`. The first run downloads the encoder.

## Reproduce

Tested with Python 3.13, 16 GB of memory, 8 cores and an RTX 2070 SUPER.

1. Download `capture24.zip` (6.9 GB) from <https://doi.org/10.5287/bodleian:NGx0JOMP5> into `data/`.
2. `pip install -r requirements.txt` and `python -m ipykernel install --user --name capture24`. Versions are pinned because the paper's pipeline depends on how imbalanced-learn 0.13.0 and scikit-learn 1.6.1 compute out-of-bag output.
3. Part 1 (CPU): `jupyter nbconvert --execute --to notebook --inplace capture24_har.ipynb`. About 7 min with caches; the first run adds about 27 min.
4. Part 2 (GPU): the same command on `capture24_new_model.ipynb`. Fits are cached in `data/fits/`; with the cache it takes about 10 min, and from scratch it trains 16 fine-tuned encoders (about 9 min each) and 61 recurrent networks (3-5 min each).
5. From the repository root, after part 2: `python scripts/train_context.py` refits the shipped model.

Code is in `scripts/`: `features.py` (windows and features), `har.py` (folds, forest, HMM, metrics, bootstraps), `deep.py` (encoder and networks), and `train_context.py` / `predict.py`. Every score is saved in `outputs/results.json` and `outputs/results_new_model.json`.

## Licence

The code is MIT. CAPTURE-24 is CC BY 4.0 (Chan et al., 2024) and is not included. The `ssl-wearables` encoder is under the University of Oxford's Academic Use Licence (non-commercial, no redistribution); it is not included and is downloaded from its authors at a pinned commit. Because the Context model and `predict.py` depend on it, they are for non-commercial use only.
