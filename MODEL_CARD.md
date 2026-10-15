# Model card: context model for 10 daily activities from a wrist accelerometer

`models/context_v1.pt`, used by `predict.py`. Built and evaluated in [`capture24_new_model.ipynb`](capture24_new_model.ipynb).

## What it does

It labels every 10 seconds of a wrist-accelerometer recording with one of 10 activities: sleep, sitting, standing, household chores, manual work, walking, mixed activity, vehicle, sports, bicycling.

```
python predict.py recording.csv.gz --out results/
```

writes a timeline CSV (one row per 10 s window, with a probability per activity) and a picture of the day as a coloured band.

## How it works

1. The recording is cut into 10 s windows and each is resized from 100 Hz to 30 Hz.
2. Each window goes through `harnet10`, the pretrained encoder of [ssl-wearables](https://github.com/OxWearables/ssl-wearables) (Yuan et al., 2024), used as downloaded and never trained on these labels. It is pinned to a commit in `deep.py` and downloaded when the tool first runs.
3. A 2-layer bidirectional GRU (128 units per direction, 1.2M parameters) reads the encoder's output for consecutive windows across the whole recording and outputs an activity per window. Only this part is in `models/context_v1.pt`.

There is no smoothing step afterwards: an HMM on top made it worse.

## Input it was validated on

- An Axivity AX3 worn on the wrist, 100 Hz, ±8 g, values in g.
- Adults recruited around Oxford, UK, wearing it for about a day of normal life (CAPTURE-24).
- Labels from wearable-camera photos 20–40 s apart plus sleep diaries, in the paper's 10-class scheme (`WillettsSpecific2018`).

Other devices, sampling rates, wear positions, populations and labelling schemes are untested. The tool stops if the samples are not 10 ms apart, and warns if the values do not look like g or go beyond ±8 g. It reads the whole recording before predicting, so it is not a real-time model.

## Training

Fitted by `train_context.py` on CAPTURE-24 participants P001–P100, with one fifth of them held out for early stopping, seed 42. This is the configuration scored below. The shipped weights are a refit of that configuration, not the file that was scored, since the scored fit's weights were not kept. The refit reproduces it: its top class is the same on 100.0% of the held-out participants' windows (checked without labels), and the tool run on one raw held-out recording gives the same timeline as the scored fit.

## Scores

Scored on labelled windows only. CAPTURE-24's annotators left an unlabelled gap at almost every change of activity, so the scored windows sit inside bouts; expect lower scores on continuously labelled data, where changes of activity are scored too.

On CAPTURE-24's 51 held-out participants (P101–P151), scored once, with 95% intervals from resampling participants:

| | κ | Macro-F1 |
|---|---|---|
| This model | 0.790 [0.758, 0.819] | 0.642 [0.577, 0.677] |
| Minus a random forest + HMM (model B of this project), paired | +0.100 [0.076, 0.124] | +0.076 [0.021, 0.111] |

The gain over B compares the rules the protocol fixed for each: this model keeps the class frequencies in its output, B has them divided out. With each at its best rule for κ in cross-validation, the gap is +0.064.

5-fold cross-validation on P001–P100, mean of three seeds: κ 0.814 and macro-F1 0.580 (a mean over folds). The cross-validation macro-F1 is the safer figure to quote; the held-out one leans on a few activities that only some participants did. The folds share most of their training participants, so cross-validation differences of 0.01–0.02 are within noise.

### Two decision rules

`--rule` chooses how much of the training class frequencies stays in the model's output, which is divided by the frequencies to the power α. Cross-validation scores:

| `--rule` | κ | Macro-F1 | What it favours |
|---|---|---|---|
| `agreement` (default, α = 0; the rule scored on the holdout) | 0.814 | 0.580 | common activities |
| `balanced` (α = 0.75, the best α for macro-F1 on the same folds) | 0.790 | 0.593 | rare activities |

### By activity (F1 on the held-out participants, default rule)

| sleep | sitting | vehicle | bicycling | household chores | sports | walking | standing | mixed activity | manual work |
|---|---|---|---|---|---|---|---|---|---|
| 0.99 | 0.93 | 0.97 | 0.85 | 0.63 | 0.60 | 0.55 | 0.35 | 0.34 | 0.21 |

The rare activities rest on few participants: in cross-validation (seed 42) sports scores 0.30 and manual work 0.01.

## Where it fails

- **Rare activities.** Manual work and mixed activity are poorly recognised (F1 0.3 or less in cross-validation), and standing is often read as sitting or household chores. Sports and bicycling scores move a lot between cross-validation and the holdout, because few participants did them.
- **Unusual days.** In cross-validation κ per participant ran from 0.24 to 0.92 for the simpler model B. One participant who spent about ten hours in a vehicle scores κ 0.39 with this model.
- **Noisy labels.** The camera labels overlap (manual work and household chores, mixed activity and walking), so some "errors" are disagreements about the label.

## Licences

- CAPTURE-24 is CC BY 4.0; credit Chan et al. (2024).
- The ssl-wearables encoder is under the University of Oxford's Academic Use Licence: non-commercial academic research only, not to be redistributed or built into a product. This repository does not include it. `predict.py` downloads it from its authors, and running the tool means accepting that licence.
- The code in this repository is MIT.

Not a medical device. Do not use it for decisions about individuals.
