# Semantic Stories for Bot Detection on TwiBot-20

Every Twitter account of the [TwiBot-20](https://github.com/BunsenFeng/TwiBot-20)
benchmark is turned into a natural-language **story** - a paragraph describing
the profile, followed by one sentence per post for the `k` most recent posts -
and pre-trained encoders are fine-tuned to read the story and classify the
account as **bot or human**. Performance is then measured as a function of `k`,
the number of posts observed (**earliness**).

The method transposes **LEGOLAS** (Pasquadibisceglie, Appice, Malerba, Fiameni),
developed for event logs, to a domain that is not an event log: the profile
fields play the role of the trace attributes, the sequence of posts the role of
the events, and the bot/human label the outcome of the case.

## A story

The profile block alone is the `k = 0` story; every post adds one sentence.
An abridged story of a bot at `k = 2`, with the handle, name, bio and post texts
replaced (the template text is exactly what the notebook produces):

> The account @example_account, named "Example", is active in entertainment
> topics. It was created six to ten years before the data was collected. […] It
> has a few hundred followers, follows a few hundred accounts, has posted tens of
> thousands of tweets in total and given tens of thousands of likes. […] It posts
> dozens of times a day and follows more accounts than follow it back. Its bio
> reads: "Just here for the memes". Most recent post: the account retweeted
> another account: "this weather is exactly what I needed [URL]". It carries 1
> link. Post 2 going back: the account retweeted another account: "New sketches
> are up, more coming this week! [URL]". It carries 1 link.

Counters are never written as digits: each one is mapped onto a logarithmic
band and rendered in English (*a few hundred followers*, *tens of thousands of
tweets*), because sub-word tokens of a number carry no ordinal meaning.

Four template variants are compared:

| Variant | Profile block | `verified` field | Behaviour block | Posts |
|---|---|---|---|---|
| `full` | yes | no | no | `k` = 0..5 |
| `full+verified` | yes | yes | no | `k` = 0..5 |
| `full+stats` | yes | no | yes | `k` = 0..5 |
| `posts+stats` | no | - | yes | `k` = 1..8 |

The **behaviour block** summarizes the observed posts in at most four sentences
(*Of its 5 most recent posts, 4 are retweets. They use no hashtag.*); which
statistics it states was decided by a per-class audit of 21 candidates.

## Experiments and results

Every experiment is run with three seeds (42, 1, 7). Per-account metrics on the
1,183 test accounts, one prediction per account at its longest prefix, mean ±
sample standard deviation over the seeds.

**Group A - which encoder** (template `full`):

| Model | Parameters | Accuracy | Macro-F1 |
|---|---|---|---|
| `roberta-base` | 124.6M | 0.7932 ± 0.0066 | **0.7897** ± 0.0083 |
| `cardiffnlp/twitter-roberta-base-2021-124m` | 124.6M | 0.7887 ± 0.0096 | 0.7857 ± 0.0088 |
| `prajjwal1/bert-medium` | 41.4M | 0.7847 ± 0.0027 | 0.7827 ± 0.0025 |
| `answerdotai/ModernBERT-base` | 149.6M | 0.7791 ± 0.0174 | 0.7746 ± 0.0185 |

**Group B - what the posts carry** (`bert-medium`):

| Template | Accuracy | Macro-F1 |
|---|---|---|
| `full` | 0.7847 ± 0.0027 | 0.7827 ± 0.0025 |
| `full+stats` | 0.7839 ± 0.0048 | 0.7831 ± 0.0046 |
| `posts+stats` (1,173 accounts) | 0.6695 ± 0.0128 | 0.6673 ± 0.0126 |

**Group C - the `verified` artifact** (`bert-medium`):

| Template | Accuracy | Macro-F1 |
|---|---|---|
| `full` | 0.7847 ± 0.0027 | 0.7827 ± 0.0025 |
| `full+verified` | 0.8281 ± 0.0086 | 0.8248 ± 0.0074 |

In short:

- **The profile carries most of the signal.** With no post at all macro-F1 is
  already 0.75-0.77; five posts add 0.02-0.03, and the curve flattens by
  `k = 4`.
- **Encoder size does not buy accuracy.** `bert-medium`, a third of the size of
  `roberta-base`, is 0.007 macro-F1 behind, within the seed spread, and the
  most stable model across seeds. In-domain pre-training on tweets does not
  help either.
- **The posts alone carry real but much weaker, largely redundant signal**
  (0.667), and a summary of the posts adds nothing on top of the profile: a
  paired McNemar test finds no difference on any seed.
- **`verified` is an annotation artifact.** It is true for 60% of the human
  test accounts and for none of the bots; the rule "verified → human, otherwise
  bot" alone reaches 0.817 accuracy. Putting the field in the story adds +0.042
  macro-F1, significant on every seed - which is why every other experiment
  leaves it out.

For orientation: RoBERTa fine-tuned on raw tweets without a template reaches
0.755 accuracy / 0.731 F1 in Feng et al. (2024). Graph-based methods exceed
0.85 but use the follow graph, which the stories do not.

The full tables, earliness curves, per-seed numbers and paired tests are in
[`RESULTS.md`](RESULTS.md).

## Repository layout

| Path | Content |
|---|---|
| [`main.ipynb`](main.ipynb) | **The project.** Data inspection, template, audits, and one block per experiment with its metrics and figures, saved with its outputs |
| `main.py` | `nbconvert` export of `main.ipynb`, for running headless |
| `main_parametrized.ipynb` | The earlier notebook, where the experiment was selected by configuration flags; kept for reference, not needed to reproduce the results |
| [`PROJECT.md`](PROJECT.md) | Method, dataset, template, design decisions with their measurements, declared limitations |
| [`RESULTS.md`](RESULTS.md) | Every number, with its reading |
| `out/results_<model>_<variant>_seed<seed>.json` | Metrics, configuration and per-account predictions of each of the 21 runs |
| `out/earliness_*.png`, `.pdf` | One earliness figure per experiment, plus one comparison per group |
| `pyproject.toml`, `uv.lock` | Environment, managed with [uv](https://docs.astral.sh/uv/) |
| `data/` | TwiBot-20 splits - not included, see below |

Not in the repository (`.gitignore`): the fine-tuned checkpoints
(`out/legolas_<model>_<variant>/`, 159-575 MB each), the intermediate trainer
checkpoints and `out/legacy/`, which holds superseded runs.

## Setup

Requirements: Python 3.14, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU
with CUDA 12.6 support. All the runs in `RESULTS.md` were made on a Quadro RTX
5000 (16 GB, Turing, fp16); `ModernBERT-base` peaks at 12.4 GB.

```bash
uv sync
```

`torch` is resolved from the PyTorch CUDA 12.6 index configured in
`pyproject.toml`.

### Data

TwiBot-20 is distributed by its authors; follow the access instructions in the
[TwiBot-20 repository](https://github.com/BunsenFeng/TwiBot-20). Place the three
official splits in `data/`:

```
data/
├── train.json   (8,278 accounts)
├── dev.json     (2,365 accounts)
└── test.json    (1,183 accounts)
```

`support.json` is not used.

## Running

```bash
uv run jupyter lab main.ipynb
```

Run the notebook top to bottom. `RETRAIN` in section 1 decides what happens to
a run whose results file already exists in `out/`:

- `RETRAIN = False` (default) - the run is loaded, not trained. With the results
  files of this repository in place, the whole notebook runs in about ten
  minutes and regenerates every table and figure, without training anything.
- `RETRAIN = True` - every run is trained from scratch: 21 runs, about six hours
  on the GPU above (~12 minutes per run for `bert-medium`, ~29 for
  `roberta-base`, ~55 for `ModernBERT-base`).

To re-train a single run, delete its `out/results_*.json` and run the notebook
with `RETRAIN = False`: only the missing run is trained.

Headless, for long runs:

```bash
uv run jupyter nbconvert --to script main.ipynb   # regenerate main.py after editing the notebook
nohup uv run python main.py > out/run.log 2>&1 &
```

## Limitations

- The stories describe an account, not its neighbourhood: the follow graph is
  not used, which is where the state of the art on TwiBot-20 gets most of its
  accuracy.
- Posts carry no timestamp in TwiBot-20, so earliness is positional ("the three
  most recent posts") rather than temporal, and no timing behaviour can be
  described.
- Stories are budgeted for a 512-token window, which caps the full template at
  five posts.
- The bot label of TwiBot-20 also covers spam and amplification accounts, and
  the dataset carries annotation artifacts (`verified` above).

The complete discussion is in [`PROJECT.md`](PROJECT.md), section 10.

## References

1. Pasquadibisceglie, V., Appice, A., Malerba, D., Fiameni, G. *Leveraging a
   large language model to predict hospital admissions of emergency department
   patients.* Expert Systems with Applications 287, 128224. **(LEGOLAS)**
2. Feng, S., Wan, H., Wang, N., Li, J., Luo, M. *TwiBot-20: A Comprehensive
   Twitter Bot Detection Benchmark.* CIKM 2021.
3. Feng, S. et al. *What Does the Bot Say? Opportunities and Risks of Large
   Language Models in Social Media Bot Detection.* ACL 2024.
4. Hegselmann, S. et al. *TabLLM: Few-shot Classification of Tabular Data with
   Large Language Models.* AISTATS 2023.
