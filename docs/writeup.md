# What a wrist sensor can and cannot tell you about someone's day

*A write-up of this repository. The numbers are from `outputs/results.json` and `outputs/results_new_model.json`.*

CAPTURE-24 is 151 people wearing a wrist accelerometer for a day each, with every stretch of the day labelled from wearable-camera photos. The task is to name the activity in each 10 seconds, out of ten: sleep, sitting, standing, household chores, manual work, walking, mixed activity, vehicle, sports, bicycling. The paper that released the data reports a random forest with a hidden Markov model (HMM) at κ 0.637 on 51 held-out participants, and a deep model at 0.735.

I set out to reproduce the first number and beat it. Three things came out of it, and the most useful one was not a model.

## 1. Reproduce first, and count

My window counts match the paper's exactly: 922,199. Getting there took one rule the published preparation script does not have, and until the counts matched, no comparison with the paper meant anything.

Rerunning the paper's pipeline from the authors' code gave κ 0.660, a little above the published 0.637. That rerun, not the published number, is the baseline everything else is paired against: same folds, same participants, same bootstrap draws.

Two evaluation pitfalls turned up on the way.

- **Splitting sequences at annotation gaps** tells the HMM where activities start and end. It is worth 0.02 macro-F1 that a deployed model would never see.
- **Out-of-bag output from a balanced forest** is missing for many samples of the big classes unless the forest is large. With 300 trees, 29% of sleep and sitting windows had none, and the library fills those rows with zeros. An HMM fitted on that lost 0.075 κ. A one-line rule predicts the share to within 0.002.

My own forest + HMM ("B") ended 0.03 κ above the rerun on the held-out participants, with an interval whose lower end is 0.001. A small gain, and I report it as that.

## 2. The labels cannot tell you how often people switch activity

Looking for where B's errors were, I counted label changes. Of 609,580 adjacent pairs of labelled windows, 46 are two different activities. Annotators leave an unlabelled gap at almost every change.

That one count explains several things at once.

- The HMM's transition matrix, counted from adjacent labelled windows, is close to "never switch". 72 of its 90 off-diagonal entries are zero, and the cost of those switches is a numerical floor I had put in to avoid log(0).
- Uncertain label timing explains little of the error (at most 0.03 κ), because the uncertain moments were already cut out as gaps.
- The HMM appears to act as a flicker filter. Sweeping an explicit switching rate over five orders of magnitude changes nothing, and errors on bouts of a few minutes do not fall at any rate. Those errors belong to the classifier.

## 3. A frozen pretrained encoder, then context

The data's authors also published a self-supervised encoder trained on 700,000 person-days of unlabelled wrist data. I used it three ways, each with settings and a selection rule written down before scoring.

- **Fine-tuned, with the HMM:** κ 0.742 on the held-out participants, +0.052 over B paired.
- **Frozen, with a small network per window and the HMM** (a control): in cross-validation, ahead of the fine-tuned model. Fine-tuning with my settings stopped improving after one or two epochs; I did not try a gentler schedule.
- **Frozen, with a recurrent network over consecutive windows:** κ 0.790 on the held-out participants, +0.100 over B paired (at different decision rules, see below), and above the paper's published deep model (an unpaired comparison with a published number). It needs no HMM.

The last comparison needed one more check. Models that keep the class frequencies in their output favour common activities and κ; models with the frequencies divided out favour rare activities and macro-F1. Comparing them at different rules mixes the model with the rule. Scoring every model along the whole range gave the honest picture:

![Macro-F1 against κ for each model as the decision rule changes](../outputs/decision_rule.png)

With each model at its best rule, Context's lead over B in cross-validation is 0.064 κ, not 0.100. To see where it comes from, I trained the same two networks on B's 39 hand-made features. The encoder and context both matter, and they overlap: the encoder beats the hand-made features by 0.070 κ without context and 0.034 with it, and context adds 0.056 on the hand-made features and 0.020 on the encoder. Context matters most when the inputs are weak. No model gets much past macro-F1 0.60.

Two caveats apply to every number here. The scored windows sit inside bouts, because the labels have a gap at almost every change of activity, so on continuously labelled data the scores would be lower. And the cross-validation folds share most of their training participants, so cross-validation gaps of 0.02 are suggestive, not established.

## What is not solved

Rare activities and unusual days; the [model card](../MODEL_CARD.md#where-it-fails) has the numbers.

## What I would tell someone starting a benchmark project

- Match the paper's sample counts before anything else.
- Rerun the baseline yourself and pair every comparison with it.
- Write the selection rule down before scoring, and count your looks at the test set. This one has been used five times, and the README says so.
- When two models disagree on which metric they win, check whether the decision rule, not the model, is deciding.
- Look at the labels as data. The most useful finding here was a count of 46.
