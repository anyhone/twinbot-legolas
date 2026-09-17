# Results

Three groups of experiments on the official TwiBot-20 splits, the ones
`main.ipynb` runs one block each, with the same hyper-parameters and the same
evaluation throughout: 3 epochs, batch 32 x 512 tokens, lr 3e-5, 10% warmup,
weight decay 0.01, fp16, best epoch selected on dev macro-F1, seeds 42, 1 and 7.
Every number below is read from `out/results_<model>_<variant>_seed<seed>.json`.

**Group A - the full template on four encoders** (sections 1-3): profile block
plus the `k` most recent posts, `k` = 0..5; 48,704 / 13,911 / 6,902 stories from
8,278 / 2,365 / 1,183 users. Seeds 42 and 1 were run on 4 September 2026, seed
7 on 16 September, on a Quadro RTX 5000.

**Group B - the ablation on `bert-medium`** (sections 4-7): `posts+stats`, a
behaviour block over the observed prefix and then the posts, no profile,
`k` = 1..8 (64,118 / 18,317 / 9,066 stories, accounts with no post drop out);
and `full+stats`, the full template with the same block between the profile
and the first post. Run on 5 and 14 September.

**Group C - the `verified` artifact** (section 8): `full+verified`, the full
template with the one profile field every other experiment leaves out. Run on
16 September.

Everything superseded - the runs of the earlier behaviour block, the results
files and exports from before the variants existed, the per-seed figures - is
in `out/legacy/`.

---

## 1. Per-user results, full template

One prediction per account, taken at the longest prefix available for that
account (1,183 accounts). This is the number comparable with published
TwiBot-20 results.

| Model | Seed | Dev macro-F1 | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|---|---|
| `roberta-base` | 42 | 0.7666 | 0.7878 | 0.7820 | 0.8177 |
| `roberta-base` | 1 | 0.7731 | 0.7912 | 0.7887 | 0.8119 |
| `roberta-base` | 7 | 0.7683 | 0.8005 | 0.7985 | 0.8187 |
| `cardiffnlp/twitter-roberta-base-2021-124m` | 42 | 0.7719 | 0.7777 | 0.7757 | 0.7969 |
| `cardiffnlp/twitter-roberta-base-2021-124m` | 1 | 0.7700 | 0.7929 | 0.7894 | 0.8165 |
| `cardiffnlp/twitter-roberta-base-2021-124m` | 7 | 0.7813 | 0.7954 | 0.7921 | 0.8183 |
| `prajjwal1/bert-medium` | 42 | 0.7710 | 0.7828 | 0.7802 | 0.8040 |
| `prajjwal1/bert-medium` | 1 | 0.7557 | 0.7836 | 0.7826 | 0.7971 |
| `prajjwal1/bert-medium` | 7 | 0.7766 | 0.7878 | 0.7852 | 0.8088 |
| `answerdotai/ModernBERT-base` | 42 | 0.7572 | 0.7591 | 0.7534 | 0.7909 |
| `answerdotai/ModernBERT-base` | 1 | 0.7629 | 0.7878 | 0.7830 | 0.8153 |
| `answerdotai/ModernBERT-base` | 7 | 0.7513 | 0.7904 | 0.7873 | 0.8127 |

Mean and sample standard deviation over the three seeds:

| Model | Params | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|---|
| `roberta-base` | 124.6M | 0.7932 ± 0.0066 | **0.7897** ± 0.0083 | 0.8161 ± 0.0037 |
| `cardiffnlp/twitter-roberta-base-2021-124m` | 124.6M | 0.7887 ± 0.0096 | 0.7857 ± 0.0088 | 0.8106 ± 0.0119 |
| `prajjwal1/bert-medium` | 41.4M | 0.7847 ± 0.0027 | 0.7827 ± 0.0025 | 0.8033 ± 0.0059 |
| `answerdotai/ModernBERT-base` | 149.6M | 0.7791 ± 0.0174 | 0.7746 ± 0.0185 | 0.8063 ± 0.0134 |

For orientation: RoBERTa fine-tuned on raw tweets, without a template, scores
0.755 accuracy / 0.731 F1 in Feng et al. 2024. Graph-based methods on TwiBot-20
are well above 0.85 accuracy, but they use the follow graph, which the stories
do not.

---

## 2. Earliness, full template

Macro-F1 on the stories in which the model had seen exactly `k` posts,
averaged over the three seeds. `k = 0` is the profile block alone.

| Posts seen | Stories | `roberta-base` | `twitter-roberta` | `bert-medium` | `ModernBERT` |
|---|---|---|---|---|---|
| 0 | 1,183 | 0.7679 | 0.7733 | 0.7596 | 0.7548 |
| 1 | 1,173 | 0.7886 | 0.7824 | 0.7764 | 0.7768 |
| 2 | 1,153 | 0.7905 | 0.7844 | 0.7810 | 0.7802 |
| 3 | 1,138 | 0.7981 | 0.7881 | 0.7854 | 0.7848 |
| 4 | 1,130 | 0.7982 | 0.7925 | 0.7909 | 0.7810 |
| 5 | 1,125 | 0.7945 | 0.7931 | 0.7899 | 0.7806 |

Gain from `k = 0` to `k = 5`, per run:

| Model | Seed 42 | Seed 1 | Seed 7 | Mean |
|---|---|---|---|---|
| `roberta-base` | +0.0274 | +0.0321 | +0.0204 | +0.0266 |
| `cardiffnlp/twitter-roberta-base-2021-124m` | +0.0205 | +0.0291 | +0.0098 | +0.0198 |
| `prajjwal1/bert-medium` | +0.0270 | +0.0334 | +0.0305 | +0.0303 |
| `answerdotai/ModernBERT-base` | +0.0129 | +0.0341 | +0.0303 | +0.0258 |

The number of stories falls from 1,183 to 1,125 because accounts with fewer
than five usable posts drop out of the longer prefixes; the rows are therefore
not scored on identical populations, and the ~5% of accounts that disappear are
the quietest ones.

Figures: `out/earliness_<model-slug>_full.png` for each experiment (seeds thin,
mean thick), `out/earliness_comparison.png` for the twelve runs on one axis.

---

## 3. Reading of the four-model comparison

**The profile block already carries most of the signal.** With no posts at all,
macro-F1 is 0.75-0.77; five posts add 0.02-0.03. The counters, the flags and
the dates of the account are what the classifier mostly relies on, and the
posts refine a decision that has largely been made.

**The gain is real but small, and it saturates immediately.** Most of it
appears with the first post (+0.017 on average); posts 2 to 5 add roughly
another +0.010, and every curve flattens or dips at `k = 4-5`.

**`roberta-base` leads, and with three seeds the lead is at the edge of the
noise.** Its average is 0.004 above `twitter-roberta`, 0.007 above
`bert-medium` and 0.015 above `ModernBERT`, against a seed spread of 0.008 for
itself and 0.019 for `ModernBERT`. Only the gap to `ModernBERT` is larger than
either model's spread; the other two are not separable.

**Encoder size does not buy accuracy here.** `bert-medium`, at a third of the
parameters (41.4M against 124.6M) and about 40% of the wall-clock time (~12
minutes per run against ~29), lands 0.007 macro-F1 below `roberta-base` - inside
the latter's own seed spread - and is by far the most stable model of the four
(± 0.0025). On this task the template and the profile fields matter, the
capacity of the encoder does not.

**In-domain pre-training does not help.** `twitter-roberta`, pre-trained on
tweets, matches `roberta-base` within noise. The stories are template English,
not raw tweet text, so its advantage on Twitter idiom applies to only a small
part of each input.

**`ModernBERT` is the weakest and the least stable**, and it costs the most
(~55 minutes per run on this GPU, since FlashAttention-2 needs Ampere). Its
8192-token window is not usable at `MAX_LENGTH = 512`, so it competes without
its one structural advantage.

---

## 4. Per-user results, posts only

One prediction per account at the longest prefix available, as in section 1, but
over **1,173 accounts rather than 1,183**: without the profile block an account
with an empty trace produces no story at all, so the 10 silent accounts of the
test split leave the evaluation entirely.

| Seed | Dev macro-F1 | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|---|
| 42 | 0.6465 | 0.6701 | 0.6671 | 0.6984 |
| 1 | 0.6599 | 0.6564 | 0.6547 | 0.6789 |
| 7 | 0.6649 | 0.6820 | 0.6800 | 0.7056 |
| **mean** | | **0.6695** ± 0.0128 | **0.6673** ± 0.0126 | **0.6943** ± 0.0138 |

The spread here is the sample standard deviation over three seeds; the table in
section 1, having two runs per model, uses half the gap between them. The two
are not the same quantity and should not be read against each other.

Against the same encoder on the full template:

| Template | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|
| profile + posts, `k` = 0..5 (2 seeds) | 0.7832 | 0.7814 | 0.8006 |
| behaviour block + posts, `k` = 1..8 (3 seeds) | 0.6695 | 0.6673 | 0.6943 |
| difference | −0.1137 | −0.1141 | −0.1063 |

The 10 accounts that leave the population do not explain that: they are 0.9% of
the test split, so even getting all of them wrong would move macro-F1 by less
than 0.01. The gap is the profile block.

The `full` runs of `bert-medium` now carry per-account predictions, so the two
templates can be compared on the 1,173 accounts they share; section 18 of the
notebook prints that paired test, and section 7 of this file reports the one
for `full+stats`, where the populations coincide.

---

## 5. Earliness, posts only

Macro-F1 on the stories in which the model had seen exactly `k` posts, mean of
the three seeds, with `bert-medium` on the full template alongside for the `k`
the two share. `k = 0` exists only for the full template: it is the profile
block alone.

| Posts seen | Stories | Posts + behaviour block | sd | Full template |
|---|---|---|---|---|
| 0 | 1,183 | — | | 0.7578 |
| 1 | 1,173 | 0.6021 | 0.0208 | 0.7734 |
| 2 | 1,153 | 0.6357 | 0.0142 | 0.7773 |
| 3 | 1,138 | 0.6528 | 0.0150 | 0.7797 |
| 4 | 1,130 | 0.6603 | 0.0182 | 0.7884 |
| 5 | 1,125 | 0.6625 | 0.0283 | 0.7881 |
| 6 | 1,119 | 0.6649 | 0.0148 | — |
| 7 | 1,118 | 0.6690 | 0.0169 | — |
| 8 | 1,110 | 0.6748 | 0.0138 | — |

Gain from `k = 1` to `k = 8`: +0.0952 (seed 42), +0.0446 (seed 1), +0.0785
(seed 7), **+0.0727 on average** - against +0.0302 for the same encoder over
`k` = 0..5 with the profile block, and +0.0740 for the template this one
replaced. As in section 2 the rows are not scored on identical populations: 63
accounts drop out between `k = 1` and `k = 8`, and they are the quietest ones.

Figures: `out/earliness_prajjwal1-bert-medium_posts+stats_seed<seed>.png` for
the single runs, `out/earliness_ablation.png` for the five `bert-medium` runs of
both templates on one axis, `out/earliness_comparison.png` for everything on
disk.

---

## 6. Reading of the ablation

**A richer behaviour block bought nothing.** The audited template - hashtag
breadth added, repetition graded instead of flagged, the `[URL]` contamination of
the register statistics removed - lands at 0.6673 macro-F1 against 0.6674 for the
version it replaced. The two runs disagree on 166 to 221 of the 1,173 accounts
depending on the seed, but symmetrically, and McNemar on each seed finds nothing:

| Seed | Only the new template is right | Only the old one is | p |
|---|---|---|---|
| 42 | 75 | 91 | 0.244 |
| 1 | 109 | 111 | 0.946 |
| 7 | 121 | 100 | 0.178 |

This is the cleanest comparison in the project so far - same encoder, same seeds,
same accounts, one difference - and it says the block's exact wording does not
matter at this level.

**What the audit did change is the operating point.** The old template's three
seeds sat at incompatible thresholds; the new one puts them within 2.4 points of
each other:

| Seed | Predicted bot | Bot recall | Human recall |
|---|---|---|---|
| 42 | 0.553 (was 0.560) | 0.707 (was 0.726) | 0.627 (was 0.635) |
| 1 | 0.529 (was 0.449) | 0.672 (was 0.599) | 0.638 (was 0.727) |
| 7 | 0.540 (was 0.425) | 0.705 (was 0.582) | 0.655 (was 0.761) |

Bot recall spans 0.035 across seeds where it used to span 0.144. That is why
bot-F1 rose (+0.021) and its spread more than halved (± 0.0138 against ± 0.0343)
while macro-F1 did not move: the classes rebalanced rather than the model getting
better. Agreement between seeds tightened too, 0.801-0.816 against 0.764-0.810.

**The posts carry real signal on their own, and it is much weaker than the
metadata.** Eight posts with a behaviour block reach 0.675 macro-F1; the profile
block alone, with no post at all, reaches 0.758. Everything the classifier can
learn from what an account writes is worth less than what it can learn from the
account's counters, flags and dates.

**The two sources are largely redundant.** Profile alone 0.758, posts alone
0.675, both together 0.788. Adding five posts to the profile buys +0.030;
adding the profile to eight posts buys +0.112. If the two described the account
independently, the combination would sit well above either of them - it does
not, so most of what the posts say is already stated by the metadata.

**Without the profile block the earliness curve does not saturate.** +0.073 from
one post to eight, still climbing at `k = 8` (+0.006 over `k = 7`), where the
full template gains +0.030 over `k` = 0..5 and is flat from `k = 4`. Two
readings fit: the textual evidence accumulates slowly and eight posts are not
enough, or the full template flattens early because the metadata has already
settled the decision. Either way the long-window experiment (ModernBERT, 8192
tokens) is far more interesting on this arm than on the full template, where
there was nothing left to gain.

**The arm is an order of magnitude less stable across seeds.** Macro-F1 sd
0.0126, against ±0.0012 for the same encoder on the full template, and the
audited block did not fix that: it fixed the threshold, not the accuracy. The
three runs still agree with each other on only 80-82% of the accounts, barely
more than their individual accuracies of 66-68%. Deprived of the metadata anchor,
each run reaches a different set of accounts.

**Dev selection tracks test better than it did, but not well.** Seed 7 now has
both the best dev macro-F1 (0.6649) and the best test macro-F1 (0.6800), where
under the old template the two were inverted. `save_run` exported it as
`out/legolas_prajjwal1-bert-medium_posts+stats/`. One seed agreeing is not
evidence the rule works at this noise level; seed 42 still has the worst dev and
the middle test score.

**What the behaviour block itself contributed is not measured.** The `posts`
control was not run, so the −0.114 against the full template mixes two changes
at once: the profile block was removed and a behaviour block was added. Nothing
in these three runs separates them.

---

## 7. The behaviour block on the full template

`full+stats` is the full template with the audited behaviour block inserted
between the profile block and the first post. The `k = 0` story has no post to
summarise, so it is byte-identical to `full`'s. `prajjwal1/bert-medium`, seeds
42, 1 and 7.

The baseline is `full` on the same three seeds. Seeds 42 and 1 are the runs of
section 1, re-evaluated from their best checkpoints; seed 7 was trained for this
comparison, with training arguments identical field by field to theirs. Both
arms carry per-account predictions for all 1,183 test accounts, so the
comparison is paired.

| Seed | Template | Dev macro-F1 | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|---|---|
| 42 | `full+stats` | 0.7577 | 0.7853 | 0.7851 | 0.7915 |
| 42 | `full` | 0.7710 | 0.7828 | 0.7802 | 0.8040 |
| 1 | `full+stats` | 0.7580 | 0.7785 | 0.7778 | 0.7904 |
| 1 | `full` | 0.7557 | 0.7836 | 0.7826 | 0.7971 |
| 7 | `full+stats` | 0.7581 | 0.7878 | 0.7862 | 0.8047 |
| 7 | `full` | 0.7766 | 0.7878 | 0.7852 | 0.8088 |

| | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|
| `full+stats` | 0.7839 ± 0.0048 | 0.7831 ± 0.0046 | 0.7955 ± 0.0079 |
| `full` | 0.7847 ± 0.0027 | 0.7827 ± 0.0025 | 0.8033 ± 0.0059 |
| difference | −0.0008 | +0.0004 | −0.0078 |

Spreads are sample standard deviations over the three seeds. The `full` mean
here, 0.7827, differs from the 0.7814 of section 1 only because seed 7 is added.

**Account by account** - McNemar on the accounts that exactly one of the two arms
classifies correctly:

| Seed | Only `full+stats` right | Only `full` right | p | Predictions changed |
|---|---|---|---|---|
| 42 | 92 | 89 | 0.882 | 181 |
| 1 | 69 | 75 | 0.677 | 144 |
| 7 | 72 | 72 | 1.000 | 144 |

Pooled over the three seeds: 233 against 236.

**Operating point:**

| Seed | Predicted bot | Bot recall | Human recall |
|---|---|---|---|
| 42 | 0.489 (`full` 0.567) | 0.753 (0.823) | 0.823 (0.735) |
| 1 | 0.516 (0.526) | 0.772 (0.786) | 0.786 (0.781) |
| 7 | 0.545 (0.569) | 0.808 (0.830) | 0.764 (0.738) |

**Earliness**, macro-F1 by prefix, mean of the three seeds:

| Posts seen | Stories | `full+stats` | `full` | Difference | Per seed (42, 1, 7) |
|---|---|---|---|---|---|
| 0 | 1,183 | 0.7496 | 0.7596 | −0.0100 | −0.0224, −0.0029, −0.0047 |
| 1 | 1,173 | 0.7645 | 0.7764 | −0.0119 | −0.0174, −0.0086, −0.0098 |
| 2 | 1,153 | 0.7723 | 0.7810 | −0.0087 | −0.0010, −0.0162, −0.0088 |
| 3 | 1,138 | 0.7807 | 0.7854 | −0.0048 | +0.0042, −0.0051, −0.0133 |
| 4 | 1,130 | 0.7868 | 0.7909 | −0.0041 | +0.0075, −0.0148, −0.0051 |
| 5 | 1,125 | 0.7899 | 0.7899 | +0.0001 | +0.0063, −0.0016, −0.0045 |

### Reading

**The block adds nothing to the full template.** Per-account macro-F1 moves by
+0.0004, one seed up and one down by about the same amount, and McNemar finds
nothing on any seed (p >= 0.68). Between 144 and 181 of the 1,183 accounts change
prediction, and they cancel out: pooled over the seeds, the accounts only one arm
gets right split 233 to 236. Section 6 found the block's *wording* irrelevant
without the profile; this finds its *presence* irrelevant with it.

**It moves the threshold, and always the same way.** On every seed `full+stats`
calls fewer accounts bots, trading bot recall for human recall. That is why
bot-F1 is lower on all three seeds (−0.0125, −0.0067, −0.0042) while macro-F1 does
not move. It is the same thing the audited block did in the posts-only arm: it
changes the operating point, not how many accounts are right.

**Short prefixes get slightly worse, including the one that did not change.**
Macro-F1 is lower with the block at `k = 0`, 1 and 2 on every seed, by about
0.01, and the gap closes by `k = 5`. The `k = 0` row is the telling one: those
stories are identical in the two arms, so the deficit cannot come from what the
model reads, only from how it was trained - a model that has learned to lean on
the block does worse where the block is missing. Dev macro-F1, which averages
over every prefix, points the same way: 0.7579 on average against 0.7677. This is
a consistent direction, not a demonstrated effect: per-prefix predictions are not
stored, so it cannot be tested account by account, and at `k = 0` one seed
carries most of the gap.

**The larger earliness gain is not an improvement.** `full+stats` gains +0.0404
from `k = 0` to `k = 5` against +0.0303 for `full`, but only because it starts
lower; both end at 0.7899.

On the full template the block costs about 20 tokens per story and buys a lower
bot-F1 and weaker short prefixes. Whatever it says about the posts, the profile
block had already said - the redundancy of section 6, measured from the other
side.

The export `out/legolas_prajjwal1-bert-medium_full+stats/` holds seed 7, the
best on dev.

---

## 8. The `verified` artifact

`verified` is true for 60.0% of the human accounts of the test split (326 of
543) and for none of the 640 bots; on the training split, 56.6% and one of
4,646. Every other experiment leaves the field out of the profile block for
that reason. `full+verified` puts it back - `bert-medium`, seeds 42, 1 and 7,
the same 1,183 accounts as `full` (A.1), one clause of difference in the
profile sentence.

| Seed | Template | Dev macro-F1 | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|---|---|
| 42 | `full+verified` | 0.8065 | 0.8183 | 0.8163 | 0.8352 |
| 42 | `full` | 0.7710 | 0.7828 | 0.7802 | 0.8040 |
| 1 | `full+verified` | 0.8147 | 0.8335 | 0.8290 | 0.8567 |
| 1 | `full` | 0.7557 | 0.7836 | 0.7826 | 0.7971 |
| 7 | `full+verified` | 0.8114 | 0.8326 | 0.8291 | 0.8536 |
| 7 | `full` | 0.7766 | 0.7878 | 0.7852 | 0.8088 |

| | Accuracy | Macro-F1 | Bot-F1 |
|---|---|---|---|
| `full+verified` | 0.8281 ± 0.0086 | 0.8248 ± 0.0074 | 0.8485 ± 0.0116 |
| `full` | 0.7847 ± 0.0027 | 0.7827 ± 0.0025 | 0.8033 ± 0.0059 |
| difference | **+0.0434** | **+0.0421** | +0.0452 |

**Account by account**, McNemar on the accounts exactly one of the two gets
right:

| Seed | Only `full+verified` right | Only `full` right | p | Predictions changed |
|---|---|---|---|---|
| 42 | 111 | 69 | 0.002 | 180 |
| 1 | 130 | 71 | < 0.001 | 201 |
| 7 | 115 | 62 | < 0.001 | 177 |

Pooled: 356 against 202. Every seed is significant on its own.

**Operating point:**

| Seed | Predicted bot | Bot recall | Human recall |
|---|---|---|---|
| 42 | 0.562 (`full` 0.567) | 0.852 (0.823) | 0.779 (0.735) |
| 1 | 0.621 (0.526) | 0.920 (0.786) | 0.731 (0.781) |
| 7 | 0.602 (0.569) | 0.902 (0.830) | 0.751 (0.738) |

**Earliness**, macro-F1 by prefix, mean of the three seeds:

| Posts seen | `full+verified` | `full` | Difference |
|---|---|---|---|
| 0 | 0.8077 | 0.7596 | +0.0481 |
| 1 | 0.8203 | 0.7764 | +0.0440 |
| 2 | 0.8273 | 0.7810 | +0.0463 |
| 3 | 0.8295 | 0.7854 | +0.0441 |
| 4 | 0.8310 | 0.7909 | +0.0401 |
| 5 | 0.8342 | 0.7899 | +0.0443 |

Gain from `k = 0` to `k = 5`: +0.0303, +0.0202, +0.0289, mean +0.0265 against
+0.0303 for `full`.

### Reading

**One field is worth +0.042 macro-F1** - more than the five posts of the full
template (+0.030), more than any difference between encoders in group A, and
it is the only change in this file that a paired test confirms on every seed.
The seeds also agree with one another far more (0.883-0.926 of the accounts,
against 0.847-0.861 for `full`): the field gives every run the same easy
answer for a third of the humans.

**The gain is flat across `k`.** It enters at `k = 0` (+0.048), where the
profile is all the model sees, and posts do not erode it: +0.040 to +0.046 at
every prefix. The artifact is not something the posts could correct.

**It is bought on the bot side.** Bot recall rises from 0.79-0.83 to
0.85-0.92 while human recall barely moves: the rule the model learns is "not
verified → bot", which is right for the 40% of humans it cannot see and wrong
for the rest.

**How much of it is the rule alone.** "Verified → human, else bot" with no
model at all scores 0.8166 accuracy on the test split; `full+verified` scores
0.8281. The field is not a feature the encoder combines with the rest, it is
close to a lookup that the rest of the template then improves by one point.
Research question 3 gets its number: reported on this dataset with `verified`
in, about a tenth of the accuracy is annotation, not detection - which is why
every other number in this file leaves it out.

**One confound, stated.** With the field the profile sentence reads `It is
verified, uses the default profile layout, …`; without it, `It is uses the
default profile layout, …`. The comparison mixes the field with that grammar
fix. The rule alone explaining most of the gap says the field is by far the
larger of the two.

The export `out/legolas_prajjwal1-bert-medium_full+verified/` holds seed 1, the
best on dev.

---

## 9. Open points

**`roberta-base` has no `_full` export.** The export that was on disk belonged
to no recorded run (it predated both runs of 4 September) and is in
`out/legacy/pre-variant/legolas_roberta-base/`; the seed-7 run scored 0.7683
on dev, below seed 1's 0.7731, so `save_run` did not write a new one. Every
other configuration has its best-on-dev seed exported.

**Per-account predictions are missing for six runs.** Seeds 42 and 1 of
`roberta-base`, `twitter-roberta` and `ModernBERT` were written before
`per_user_predictions` existed; their best checkpoints are in
`out/legacy/<model>_seed<seed>/`, so the recovery used for `bert-medium`
(evaluation only, gated on reproducing every stored metric exactly) would add
them without re-training. Until then no paired test between encoders is
possible, and the group A comparison is a comparison of averages.

**Per-class results are stored only as predictions.** The results files keep
accuracy, macro-F1 and bot-F1; per-class recall is recomputed from
`per_user_predictions` where it exists, as sections 7 and 8 do.

**The `posts` control has not been run.** Three runs of `bert-medium` on the
`posts` variant, ~48 minutes, are what separates the effect of dropping the
profile block from the effect of adding the behaviour block. Until then section
6's last point stands: the ablation measures the pair of changes, not either
one. The variant is not in `main.ipynb`; `main_parametrized.ipynb` still builds
it.

**`out/` holds 41 GB, 21 of them in `out/legacy/`.** The intermediate
checkpoint directories dominate both. The ones under `legacy/` are the only
copies of the six best checkpoints the point above needs; the ones at the top
level (`<model>_<variant>_seed<seed>/`) can go, every run of theirs has its
results file and, where it won on dev, its export.
