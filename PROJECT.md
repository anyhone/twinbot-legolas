# Semantic Stories for Bot Detection on TwiBot-20

Preliminary project documentation. University exam project: applying the
*semantic stories* technique — narrative verbalization of structured data
followed by fine-tuning of a pre-trained encoder — to binary classification on
social media data.

---

## 1. Objective

Classify Twitter accounts as **bot** or **human** by turning each account into a
natural-language story, then fine-tuning a pre-trained encoder on those stories.

The goal is not to beat the state of the art on TwiBot-20 — the leading methods
use the follow graph, which is not used here. The goal is to check whether the
semantic-stories method, developed for process mining, transfers to a different
domain, and to measure **how much content must be observed** before an account
can be classified correctly (*earliness* analysis).

### Research questions

1. Do semantic stories work on data that is not an event log?
2. How many posts from an account are needed to classify it? Does performance
   saturate?
3. How much of the performance comes from features that are annotation
   artifacts of the dataset rather than genuine signal?

---

## 2. Origin of the method

The technique follows **LEGOLAS** (Pasquadibisceglie, Appice, Malerba, Fiameni,
*Leveraging a large language model to predict hospital admissions of emergency
department patients*, Expert Systems with Applications 287, 128224), presented
in class.

Original pipeline:

1. Traces (ordered sequences of events) are extracted from an **event log**.
2. For each trace, **prefixes** of length 1..k are generated.
3. Each prefix is converted into a natural-language **story** via a narrative
   template, labelled with the outcome of the case.
4. An encoder LLM (BertMedium, ClinicalBert, GPT2, RoBERTa) is **fine-tuned** on
   the stories.
5. Accuracy is evaluated **as a function of prefix length** (earliness) and
   explained with Integrated Gradients.

Complementary methodological reference: **TabLLM** (Hegselmann et al., AISTATS
2023), which serializes tabular rows into natural language; in their
experiments the "Text Template" serialization — a textual enumeration of the
features — performs best.

---

## 3. Dataset

**TwiBot-20** (Feng, Wan, Wang, Zheng, Luo, *TwiBot-20: A Comprehensive Twitter
Bot Detection Benchmark*, CIKM 2021).

### Structure

Each record is a user with:

| Field | Content |
|---|---|
| `ID` | Twitter identifier |
| `profile` | 38 Twitter API fields (counters, flags, dates, colours) |
| `tweet` | list of the ~200 most recent tweets, **text only** |
| `neighbor` | sampled follower and following IDs |
| `domain` | list drawn from politics, business, entertainment, sports |
| `label` | `"1"` = bot, `"0"` = human |

### Splits

Official dataset splits, **not recomputed**, to stay comparable with published
work.

| Split | Users | Stories generated (k_max = 5) |
|---|---|---|
| train | 8,278 | 48,704 |
| dev | 2,365 | 13,911 |
| test | 1,183 | 6,902 |

Train balance: 4,646 bots / 3,632 humans.

`support.json` (5.27 GB) is **not used**: it holds unlabelled users from the
third BFS level, with `neighbor` always null. It is only relevant to
semi-supervised or graph-based approaches.

### Checks performed

- **No ID overlap** between train, dev and test (verified: 0 for all three
  pairs). This matters because every prefix of a user shares the same profile
  block and the same label; splitting at the prefix level, or using overlapping
  splits, would leak.

---

## 4. Adapting the method to the dataset

TwiBot-20 **is not an event log**. This is the main deviation from LEGOLAS and
must be stated explicitly.

### What plays the role of a trace

A user's ~200 tweets are treated as a sequence of events; the profile is a trace
attribute (analogous to "Patient information" in LEGOLAS). Prefix *k* is the
**k most recent posts**.

### Limitation: no per-post timestamp

The `tweet` field is a flat list of strings: no `created_at`, no IDs, no
per-tweet metrics. Consequences:

- **Inter-post delays cannot be computed.** Publication regularity — likely the
  most discriminative signal for a bot — is not recoverable at the level of an
  individual post. It is only recovered in aggregate form, as
  `statuses_count / account age`.
- **Earliness is positional, not temporal.** "k=3" means "the 3 most recent
  posts have been observed", not "a certain amount of time has elapsed". This is
  the honest phrasing and should be used in every description of the results.

### Ordering assumption

The tweets are assumed to be returned by the Twitter API in reverse
chronological order (most recent first). The observed ordering is consistent
with this, but it is **not documented** in the dataset.

### Prefix definition

`stories[k] = profile + first k posts`, for `k` from 0 to `min(n_tweets, k_max)`.

Two intended properties:

- Prefixes are **nested** (`k ⊂ k+1`).
- The definition **does not depend on `k_max`**: prefix 1 is always the most
  recent post, whatever limit is chosen. (An earlier implementation reversed the
  list before slicing, which broke this property; it has been fixed.)

**Prefix 0** — profile only, no posts — is included deliberately: it acts as a
free metadata-only detector inside the same model, and it recovers the 55
training users that have no tweets.

---

## 5. The template

### Profile block (trace attributes, always present)

```
The account @{screen_name}, named "{name}", {DOMAIN}.
It was created {AGE} before the data was collected.
It is {verified|not verified}, {layout}, [{default picture},] {geo}, {url}.
It has {FOLLOWERS}, follows {FRIENDS}, has posted {STATUSES} in total
  and given {FAVOURITES}. It appears in {LISTED}. It {RATE} and {RATIO}.
Its bio reads: "{description}".        ← otherwise: It has no bio.
Declared location: {location}.         ← omitted when absent
```

### Event block (repeated k times)

```
{POSITION}: the account {ACTION}: "{text}".[ It carries {META}.]
```

- `POSITION`: `Most recent post` (i=1) / `Post {i} going back`
- `ACTION`: `posted` / `retweeted another account` /
  `retweeted its own alternate account`
- `META`: non-zero components only, correctly pluralized
  (`2 hashtags, 1 mention, 2 links`); the whole clause is dropped when all
  three are zero.

### Behaviour block (variants with `post_stats`, right before the events)

Emitted only when the variant sets `post_stats`: in `posts+stats`, where it opens
the story, and in `full+stats`, where it sits between the profile block and the
first post. It is the section 6 treatment - derive a value, render it as English
- moved from the profile fields to the observed posts. It needs at least one
post, so a `k = 0` story of `full+stats` is identical to one of `full`.

```
Of its {k} most recent posts, {none is a retweet | one is a retweet |
  {n} are retweets | all are retweets}.
[Two of them are nearly identical. | Its posts closely resemble one another.]
They use {no hashtag | a single hashtag | {n} different hashtags}.
[They are {written mostly in a non-Latin script}[, {rich in emoji}]
  [ and {heavily capitalized}].]
```

At `k = 1` every sentence takes the singular: `Its most recent post is [not] a
retweet. It uses no hashtag. It is rich in emoji.`

Three rules decide what it may say, and each one is load-bearing:

- **Computed on `observed_posts[:k]`, never on the whole trace.** A story at
  `k = 3` describes three posts. Summarising all 200 would leak text the model
  has not been shown and would empty the earliness curve of its meaning.
- **Placed before the events.** At the top of the token budget truncation eats
  the tail, so a summary put last would be the first thing lost.
- **Measured on the cleaned, truncated bodies the story displays**, never on the
  raw posts. A block stating something the reader cannot check is unverifiable
  from the story, and section 6.6 shows what that rule threw out.

| Sentence | Fires when | Constant |
|---|---|---|
| `{n} are retweets` | always, counting the prefix | - |
| `Two of them are nearly identical.` | max pairwise 3-gram Jaccard >= 0.9 | `NEAR_DUPLICATE` |
| `Its posts closely resemble one another.` | mean pairwise Jaccard >= 0.12, and the sentence above did not fire | `REPETITIVE` |
| `{n} different hashtags` | always, counting distinct hashtags visible in the story | - |
| `written mostly in a non-Latin script` | non-Latin share of the letters >= 0.5 | `NON_LATIN_SHARE` |
| `rich in emoji` | at least one emoji per post | `EMOJI_PER_POST` |
| `heavily capitalized` | upper-case share of the placeholder-free text >= 0.10 | `UPPERCASE_SHARE` |

### Verbalizations

| Slot | Values |
|---|---|
| AGE | less than three months · less than a year · one to three years · three to six years · six to ten years · more than ten years |
| Counters | no · fewer than a hundred · a few hundred · a few thousand · tens of thousands of · hundreds of thousands of · millions of |
| RATE | posts very rarely · posts less than once a day · posts a few times a day · posts dozens of times a day · posts at an extremely high rate |
| RATIO | follows far more accounts than follow it back · … · has vastly more followers than it follows |
| DOMAIN | `is active in {x} topics` / list / `is active across politics, business, entertainment and sports` (all four domains) |

---

## 6. Design decisions and rationale

### 6.1 Logarithmic binning of counters

**Decision:** all numeric counters are verbalized into logarithmic bins, never
passed as digits.

**Rationale:** the WordPiece tokenizer splits `1247893` into sub-tokens carrying
no ordinal meaning; the model cannot learn that a million exceeds ten thousand.
This mirrors the discretization of *numerical views* in JARVIS, applied before
encoding.

Bin coverage was checked on the training set: no empty bin, none absorbing a
dominant share (only the tail bins, "millions of", have few cases).

### 6.2 Discarded fields

- **Profile colours, image URLs, `utc_offset`, `time_zone`, `lang`**: noise;
  `lang` is deprecated and always null.
- **`protected`**: zero variance (0.0 in both classes). It costs tokens and
  carries no information.

Feng et al. (ACL 2024) use five metadata fields — follower count, following
count, tweet count, verified, active years — as the ones they consider most
useful for identifying bots. The selection here is compatible, with behavioural
flags added.

### 6.3 Tweet text handling

- URLs replaced with `[URL]`; the link count becomes an explicit feature.
- The `RT @handle:` prefix is stripped and turned into `ACTION` — the
  information is structural, not textual.
- Whitespace normalized.
- Truncation at **150 characters** per post.
- Profanity and slang are **not cleaned**: they are register signal.

The 150-character cap was chosen empirically. Lowering it from 220 to 150 moved
the median story length only from 418 to 413 tokens — most tweets are already
below the cap — but it brought the 95th percentile down from 560 to 527,
halving the number of truncated stories.

### 6.4 k_max = 5

**Decision:** at most 5 posts per story.

**Rationale:** BERT's 512-token limit. Share of stories exceeding it, measured on
the training set with the `bert-base-uncased` tokenizer:

| k | median tokens | p95 | over 512 |
|---|---|---|---|
| 0 | 148 | 180 | 0.0% |
| 3 | 280 | 348 | 0.7% |
| 5 | 371 | 468 | **1.9%** |
| 6 | 413 | 527 | **7.4%** |

At k=6, one story in thirteen loses its last post — precisely what would
distinguish prefix 6 from prefix 5 — so the earliness curve would flatten
through a truncation artifact rather than genuine saturation. At k=5, 1.9% is
negligible.

### 6.5 Features measured and dropped

- **Self-retweet** (retweeting one's own alternate handle): 371 occurrences out
  of 48,365 posts (0.77%), with near-identical rates across classes (0.66% for
  humans vs 0.85% for bots). The branch remains in the code but carries no
  signal.
- **Retweet rate as a trace attribute**: 26.1% for humans vs 39.0% for bots. A
  real but modest difference, and the information is already readable from the
  repetition of `retweeted another account` across event blocks. Not added to
  the profile block, to avoid spending tokens on redundancy. In the posts-only
  variants, where there is no profile block and the retweet rate is the single
  most discriminative statistic available, it opens the behaviour block instead
  (section 6.6); `full+stats` carries that block after the profile.

### 6.6 What the behaviour block says, and why not more

Every candidate was scored by Cohen's *d* on the pooled standard deviation over
the training split, on an 8-post window (the notebook uses `min(k_max, 8)`, so 5
posts for the variants capped at `k = 5`), so the numbers stay comparable between
arms. **Section 8 of the notebook recomputes
this whole table on every run**, so the justification cannot drift away from the
code that ships.

A statistic has to clear two bars.

**Bar 1 - it has to separate the classes**, |*d*| >= 0.12. Below that the
sentence costs tokens the posts need and returns noise. Three things about that
number, since it is a threshold this project chose rather than one it inherited:

- **It is not a convention of the literature.** Cohen's own scale calls 0.2
  small, 0.5 medium and 0.8 large, so 0.12 sits below even "small". The bar is
  deliberately permissive, because the question it answers is not "is this
  effect notable" but "is this sentence worth five to twenty tokens of a
  512-token budget".
- **Statistical significance is useless as a filter at this sample size.** The
  audit population is 3,592 humans and 4,631 bots, and with those numbers
  |*d*| = 0.044 already reaches p < 0.05 (0.057 reaches p < 0.01). A
  significance test would admit 12 of the 21 candidates, `post-length spread`
  (−0.047) and `link share` (−0.045) among them: what makes them significant is
  the sample size, not the size of the difference.
- **The value sits at a break in the measured distribution.** Sorted by |*d*|,
  the candidates fall into a group of nine at 0.118 and above, then a gap of
  0.021, then a tail that decays quickly - 0.097, 0.076, and nothing else above
  0.047. The bar is drawn at that break: data-driven, but chosen after seeing
  the numbers, and a bar at 0.08 would have been equally defensible - it would
  have admitted one further statistic, `digit share`.

**How much the exact value matters: very little**, and that is worth stating
rather than hiding. The three largest rejections of the audit (+0.35, +0.26,
+0.21) fail bar 2, not bar 1; the bar was deliberately overridden once, for the
sharp repetition sentence at +0.118; and group B of `RESULTS.md` measured the
whole block as worth +0.0004 macro-F1 on the full template, with McNemar finding
no difference on any seed. Any cut between 0.08 and 0.15 would have produced the
same template and the same conclusions. What bar 1 really does is keep the tail
of the candidate list out of the story; the decisions that shaped the block were
made by bar 2.

**Bar 2 - it has to say something new.** Statistics come in correlated families,
and a family gets one sentence, not three. Four candidates with a large *d* fail
here, and they are the instructive part of the audit:

| Rejected candidate | *d* | Why |
|---|---|---|
| distinct retweet sources | +0.35 | *r* = 0.94 with the retweet count, and at a fixed number of retweets the gap between the classes is about zero. Bots do not retweet *more accounts*, they retweet *more*. |
| posts ending in an ellipsis | +0.26 | 57.4% of retweets end in one against 0.0% of the account's own posts; *r* = 0.77 with the retweet share. It is the retweet count wearing a different hat. |
| distinct accounts mentioned | +0.21 raw, **+0.08 visible** | The signal sits in the `RT @handle:` prefixes that `clean_tweet` removes. Measuring the block on raw posts instead of visible bodies would have shipped a sentence the reader cannot check - this is the case the third rule of section 5 exists for. |
| hashtags per post | +0.17 | *r* = 0.89 with the distinct count, which separates better (+0.18). |

What survives is one sentence per family:

| Family | Statistic | *d* | Rendered as |
|---|---|---|---|
| retweeting | share of the prefix that is a retweet | **+0.39** | `Of its 8 most recent posts, 3 are retweets.` |
| script | non-Latin share of the letters | −0.26 | `written mostly in a non-Latin script` |
| hashtags | distinct hashtags visible in the story | +0.18 | `They use 4 different hashtags.` |
| capitalization | upper-case share | +0.17 | `heavily capitalized` |
| repetition, diffuse | mean pairwise 3-gram Jaccard | +0.16 | `Its posts closely resemble one another.` |
| repetition, sharp | max pairwise 3-gram Jaccard | +0.12 | `Two of them are nearly identical.` |
| emoji | emoji per post | −0.13 | `rich in emoji` |

Two of the seven are **human** tells, not bot tells: a non-Latin script fires on
4.6% of humans against 1.0% of bots (lift 0.21) and emoji on 17.0% against 11.3%
(lift 0.66). They are kept because a statistic that separates the classes is
useful in whichever direction it points.

The sharp repetition sentence is the one deliberate exception to bar 1: *d* =
+0.118, marginally under, but its lift of 1.60 is the highest of any sentence
except `all are retweets`, and it is the sharp end of a family whose diffuse end
clears the bar.

**Bar 2 against the profile.** The audit above was run on posts-only stories.
Once the block is added to the full template (`full+stats`), saying something new
also means not restating the profile block, so every surviving statistic was
correlated with the profile fields the template verbalizes: followers,
following, statuses, likes given, account age, posting rate, default layout,
geolocation and website. Over 8,223 training accounts on a 5-post window the
strongest correlation is |*r*| = 0.22, retweet share against likes given; for
every other statistic of the block the strongest is at most 0.15.

**Dropped for |*d*| < 0.12**, with their values: digit share (+0.10), distinct
mentions visible in the story (+0.08), post-length spread (−0.05), link share
(−0.05), hashtag repetition (+0.04), reply share (+0.04), type-token ratio
(−0.03), mean post length (−0.03), self-retweet share (+0.02), exclamations per
post (+0.02).

**Not measurable on this dataset at all**, which is the ceiling on how rich the
block can get:

- *posting times and engagement*: `tweet` is a list of plain strings, with no
  timestamp and no like or retweet count attached to a post. Bursts, regular
  intervals and hours of the day - the classic behavioural bot tells - are
  simply absent from the data.
- *link destinations*: 3,856 of 3,904 URLs in a sample are `t.co` shortlinks, so
  link diversity cannot be measured.
- *the follow graph*: `neighbor` is null in this dump.

Two corrections this audit made to the first version of the block, both found by
reading generated stories rather than by looking at aggregates:

- the register statistics were counting the `[URL]` placeholder, which is
  upper-case. 45% of the `heavily capitalized` firings were caused by it (344 of
  759 accounts), and they leaned toward bots because bots post more links - the
  sentence was silently saying "posts mostly links". Now measured on
  placeholder-free text.
- `They are ...` was emitted in the plural even for a single post.

---

## 7. The `verified` artifact

On the training set, `verified` is True for **56.6% of humans** and for **0.0%
of bots** (0 out of 4,646). The TwiBot-20 annotators most likely used the blue
check as a heuristic for labelling humans.

The trivial rule "verified → human" correctly classifies 2,057 of 8,278 accounts
without looking at anything else.

**Decision:** the field is left out of every template. One experiment puts it
back - `full+verified`, group C of the notebook: the full template on
`bert-medium` with `verified` in the profile sentence, against `full` on the
same seeds and accounts. That gap measures how much of the performance is
artifact.

The dataset authors themselves, when introducing TwiBot-22, cite low annotation
quality as a limitation of earlier benchmarks.

**Consequence for interpretation:** the "bot" label in TwiBot-20 also covers
spam and amplification, not automation alone. The model learns that operational
definition, not "automated account".

---

## 8. Parsing pitfalls

Documented because they are not obvious and produce silent failures.

1. **Every value is a string with a trailing space.** `"13324 "`, `"False "`,
   `"1"`. Call `.strip()` before every cast.
2. **`bool("False ")` is `True`.** Boolean casting must compare against
   lowercase `'true'`. Sanity check: if every account comes out verified, the
   cast is broken.
3. **`profile_location` and `entities` are serialized Python dicts**, not JSON:
   single quotes and `None`. `json.loads` fails; `ast.literal_eval` is needed.
   (These fields are not used anyway.)
4. **`domain` is a list** and may hold several values: 7,125 users have a single
   domain, 1,153 have more than one, and 253 have all four.
5. **`created_at`** uses the format `%a %b %d %H:%M:%S %z %Y`. Account age must
   be computed against the **collection date (2020)**, not against today.
6. **Traces are shorter than 200** for many users; 55 in the training set have
   no tweets at all (with a median `statuses_count` of 0 — genuinely inactive
   accounts).

---

## 9. Experimental setup

| | |
|---|---|
| Models | `roberta-base` (124.6M, reference), `prajjwal1/bert-medium` (41.4M, cost of shrinking the encoder), `cardiffnlp/twitter-roberta-base-2021-124m` (124.6M, effect of in-domain pre-training), `answerdotai/ModernBERT-base` (149.6M, the only 8192-token window) |
| Tokenizer | each model uses its own, except bert-medium, which ships none: `bert-base-uncased` (identical 30522 WordPiece vocabulary). twitter-roberta shares RoBERTa's BPE exactly, so the two see byte-identical token sequences |
| Max length | 512, with truncation |
| Story variants | `full` (profile block + posts, k = 0..5), `full+stats` (profile block + behaviour block + posts, k = 0..5), `posts` (posts only, k = 1..8), `posts+stats` (posts only plus the behaviour block, k = 1..8), `posts+stats-long` (the same on a geometric grid up to k = 64, 8192 tokens). One notebook run builds one variant and stamps its name on every artifact it writes |
| Seeds | 42, 1 and 7. One run per (model, variant, seed); the four-model comparison used 42 and 1 only, and its numbers are reported as such |
| Batch | 32 for every model, so the comparison measures the encoder. Peak memory at 32 x 512 tokens in fp16, of the 15.5 GiB usable on a Quadro RTX 5000: 7.9 GiB roberta-base, 8.1 twitter-roberta, 12.4 ModernBERT; throughput is already saturated at 32. On out-of-memory, batch 16 with `gradient_accumulation_steps=2` keeps the effective batch and halves the footprint |
| LR | 3e-5 / 2e-5, 10% warmup, weight decay 0.01, fp16 |
| Selection | `load_best_model_at_end` on dev `macro_f1` |
| Metrics | Overall Accuracy, macro-F1, weighted-F1 (as in LEGOLAS) |

### The posts-only ablation

The four-model comparison showed the profile block carrying most of the signal:
macro-F1 is already 0.75-0.76 with no posts at all. That measures what the posts
add on top of the metadata, not what they carry on their own, so the `posts` and
`posts+stats` variants drop the profile block entirely and start the prefixes at
k = 1 (an empty prefix would be an empty story; the 10 test accounts with no post
disappear from the arm).

Dropping the block frees the ~150-200 tokens it occupies, which is what lets `k`
grow inside the same 512-token window. `K_MAX = 8` comes from the token audit,
not from a guess: the share of stories that overflow 512 tokens is 4.0% at k = 8
(5.1% with the behaviour block) under WordPiece and 5.8% (6.4%) under RoBERTa's
BPE, which brackets the 4.4% the full template already accepts at k = 5. At k = 9
it is 9.5%, and by k = 12 it is 58%. The binding constraint is the p95 of the
length, not its median: a median story at k = 8 is only 356 tokens long.

The ablation runs on `prajjwal1/bert-medium` alone. The model comparison put it
0.004 macro-F1 from `roberta-base` - well inside the seed spread - while being
the fastest and the most stable across seeds, so it is the sensible workhorse for
a grid. The winning arm can be re-run on `roberta-base` for a final table.

### The long-window arm

`posts+stats-long` answers the question the 8-post arm left open. That arm gained
+0.074 macro-F1 from `k = 1` to `k = 8` and was **still climbing at the last
point**, where the full template gained +0.030 over `k = 0..5` and was flat from
`k = 4`. Nothing in those numbers separates a real plateau from a 512-token
budget running out, and only a longer window can, so this arm runs on
`answerdotai/ModernBERT-base` and its 8192-token context - the one structural
advantage that model never got to use in the four-model comparison.

The data supports it. 86.5% of the training accounts have at least 64 posts and
the median account carries the full 200-post trace. At `k = 64` a story measures
2,750 ModernBERT tokens at the median and 4,002 at the p95; the ceiling only
starts to bite at `k = 150`, where the p95 reaches 9,303.

What does not support it is the nested-prefix scheme, quadratic in `k_max`:
every prefix length from 1 to 32 would cost 246k training stories against the
64k of the 8-post arm. Hence the `prefixes` key - a geometric grid
`1, 2, 4, 8, 16, 32, 64` costing **54,452 training stories, fewer than the arm it
extends**, over a range eight times wider. Two consequences worth stating:

- an account is read at the largest grid point its trace reaches, so 8.7% of the
  test accounts are scored a median of four posts short. The alternative - adding
  each account's own length to the grid - costs only 103 extra test stories but
  introduces 45 additional `k` values whose median bucket holds two stories,
  putting noise on the one curve the arm exists to measure;
- every account with at least one post still produces a `k = 1` story, so the
  per-account population is the same 1,173 as the other posts-only arms, and the
  per-user numbers stay directly comparable.

Measured cost on the Quadro RTX 5000, `batch_size = 2` (batch 4 and above run
out of memory on a batch that happens to contain several long stories, so 2 is
not conservative, it is the ceiling):

| Grid | Train stories | Mean tokens | ms/story | Peak GPU | 3 epochs |
|---|---|---|---|---|---|
| 1..64 | 54,452 | 823 | 118 | 9.5 GiB | **5.3 h/seed** |
| 1..32 | 47,294 | 493 | 68 | 9.0 GiB | 2.7 h/seed |
| 1..16 | 39,898 | 305 | 41 | 3.8 GiB | 1.3 h/seed |

Only 0.21% of the stories reach the 8192-token ceiling, so the window is not the
binding constraint any more - the wall clock is. Building the stories costs a
further ~5 minutes per run: `pair_similarities` is quadratic in `k` and is
recomputed at every prefix, which is 2,667 pair comparisons per account against
84 for the contiguous 8-post grid.

`STORY_VARIANTS` therefore carries `max_length` and `batch_size` per arm, and
`EFFECTIVE_BATCH` keeps the optimizer identical across arms through gradient
accumulation: a long window changes the sequence length and nothing else.
`fine_tune` refuses to start when the selected encoder's window is smaller than
the variant's, which is the one failure mode a grid like this cannot afford -
`bert-medium` on the long arm would silently drop every post past the first few.

### Notes on `transformers` 5.x

- `warmup_ratio` has been **removed**; `warmup_steps` accepts a float,
  interpreted as a fraction of total steps.
- `evaluation_strategy` → `eval_strategy`.
- In `Trainer`, `tokenizer` → `processing_class`.
- `prajjwal1/bert-medium` does not declare `model_type` in its `config.json`, so
  `Auto*` classes fail; the config must be built explicitly with `BertConfig`
  (hidden 512, 8 layers, 8 heads, intermediate 2048). It also ships no
  serialized tokenizer, only a `vocab.txt`, so `AutoTokenizer` fails on it too.

### Notes on the GPU (Turing, sm_75)

- bf16 is only emulated: training runs in fp16 with the `GradScaler`, which
  skips two or three updates at the very start while it calibrates its scale.
- FlashAttention-2 requires Ampere or later, so `ModernBERT-base` falls back to
  SDPA and loses its speed advantage: ~47 samples/s against 84 for
  `roberta-base`, i.e. ~45 minutes per run against ~25.
- `vinai/bertweet-base` was considered and rejected: `max_position_embeddings`
  is 130, so it would see the profile block and almost nothing else.
- `microsoft/deberta-v3-base` needs `sentencepiece` (or `tiktoken`) installed
  before its tokenizer can be loaded at all.

---

## 10. Declared limitations

1. **No per-tweet timestamps** → earliness is positional, not temporal.
2. **Reverse chronological order assumed**, not documented.
3. **Seed variability** is of the same order as the differences between adjacent
   k values, so single-run differences below one percentage point are not
   interpretable. Multiple seeds per configuration are required.
4. **No graph information**: `neighbor` is used for counts only.
5. **Asymmetric truncation across models**: at k=5 the share of stories pushed
   past 512 tokens is 4.4% for RoBERTa's BPE (and for twitter-roberta, which
   shares it), 3.3% for ModernBERT and 2.4% for WordPiece, so truncation
   penalizes the RoBERTa family slightly more than the others. It also means
   `K_MAX = 5` is a budget of the window, not a property of the method. Two
   things buy room: dropping the profile block, which takes `K_MAX` to 8 at the
   same truncation share, and ModernBERT's 8192-token context, which the
   `posts+stats-long` variant takes to 64 on a geometric prefix grid. Built,
   not yet run.
6. **Profile block repetition**: each user appears once per prefix in the
   training set with ~150 identical tokens. This is not test leakage — the
   splits are disjoint — but it encourages memorization.
7. **Operational definition of "bot"** inherited from the dataset, covering spam
   and amplification.

---

## 11. Code structure

`main.ipynb`, run top to bottom: sections 1-10 build the stories of every
variant once, sections 11-14 define the pipeline as functions, sections 15-20
are the experiments - one block each, model, template and seeds written out -
with a comparison closing every group. A block whose results file exists loads
it instead of training (`RETRAIN = False`), so the whole notebook regenerates
every table and figure in minutes and trains only what is missing.

| Section | Content |
|---|---|
| 1-2 | Setup (`DATA_DIR`, `SEEDS`, `RETRAIN`), GPU check |
| 3-5 | Loading the training split, `parse_user`, diagnostics per class |
| 6 | Verbalization functions + band coverage |
| 7 | `STORY_VARIANTS`, `clean_tweet`, `profile_block`, `post_stats_block`, `event_block`, `build_stories(user, variant)`; the training stories of every variant |
| 8 | Per-class audit on a fixed 8-post window: the retweet branches, every sentence of the behaviour block, the Cohen's d table of every candidate statistic |
| 9 | `MODEL_SPECS` (tokenizer + constructor per model), `MAX_LENGTH`, token-length audit per variant |
| 10 | dev/test through the same pipeline, `datasets[variant][split]`, `tokenize_splits(variant, tokenizer)` |
| 11-13 | `fine_tune`, `evaluate_on_test`, `save_run` - every one takes the variant |
| 14 | `run_experiment` (load or train, one run per seed) and `report_experiment` (per-seed table, earliness figure) |
| 15-16 | Group A: `full` on `bert-medium`, `roberta-base`, `twitter-roberta`, `ModernBERT`; comparison |
| 17-18 | Group B: `posts+stats` and `full+stats` on `bert-medium`; comparison with `full`, including the paired McNemar test on the shared accounts |
| 19-20 | Group C: `full+verified` on `bert-medium`; paired comparison with `full` |

**Configuration** (section 7): `STORY_VARIANTS` maps a name to its three
switches - `profile`, `post_stats`, `verified` - and `k_max`; every function
downstream takes the variant name or dict as an argument, there is no selected
variant. Shared constants: `MAX_TWEET_CHARS`, `MAX_BIO_CHARS`, the thresholds of
the behaviour block (`NEAR_DUPLICATE`, `REPETITIVE`, `UPPERCASE_SHARE`,
`EMOJI_PER_POST`, `NON_LATIN_SHARE`, calibrated in section 8), `MAX_LENGTH` and
the seeds. A new experiment is one markdown cell and two code cells in the
shape of sections 15-20; a new model is an entry in `MODEL_SPECS`.

**Artifacts**, all under `out/`, every name carrying the variant so no two
experiments overwrite one another: `results_<model>_<variant>_seed<seed>.json`
(metrics, configuration and per-account predictions of one run - the file
`run_experiment` loads), `legolas_<model>_<variant>/` (the exported checkpoint,
kept for the best seed by dev macro-F1), `<model>_<variant>_seed<seed>/` (the
Trainer's intermediate checkpoints), `earliness_<model>_<variant>.png|pdf` (one
figure per experiment, seeds and mean), `earliness_comparison.png|pdf` (group
A), `earliness_ablation.png|pdf` (group B) and `earliness_verified.png|pdf`
(group C).

The runs written before the variants existed were migrated to these names:
their results files gained the `variant`, `include_profile` and
`include_post_stats` keys, the two `bert-medium` runs were re-evaluated from
their best checkpoints to add per-account predictions (every stored metric
reproduced exactly), and the exports of `twitter-roberta` and `ModernBERT` were
renamed `_full`. Every original is in `out/legacy/pre-variant/`, together with
`legolas_roberta-base/`, an export that belongs to no recorded run. The
parametrised notebook those runs came from is kept as
`main_parametrized.ipynb`; `main.py` is the `nbconvert` export of `main.ipynb`.

---

## 12. Refactoring notes

Resolved in the current version: the per-model branch that duplicated the
training cells is gone (the model is an entry in `MODEL_SPECS`), seeds are a
loop rather than a hand edit, every run persists to `out/results_*.json` instead
of being overwritten in memory, the token-length audit samples with a fixed
seed, the template is a named variant rather than a set of flags that can
disagree, and every artifact name carries that variant.

Still open:

1. ~~Global state~~ - resolved: the variant is an argument of every template
   and pipeline function, and `verified` is one of its switches.
2. **`build_stories` mixes three responsibilities**: verbalization, prefix
   generation and output format. These should be separated.
3. **No tests.** Parsing at least — boolean casts, dates, multi-valued domains —
   deserves unit tests, given the pitfalls in section 8.
4. **The comparison plots one line per run**, so several seeds of one
   configuration read as separate series. Aggregating over seeds (mean with a
   min-max band) is what the results table would need.
5. **`main.py` is a stale `nbconvert` export** of an earlier version of the
   notebook and is not regenerated by anything.
6. ~~One variant per notebook run~~ - resolved: the stories and datasets of
   every variant are built once in sections 7 and 10, and each experiment names
   its variant.
7. **Per-variant populations are reconciled by hand.** The results files carry
   `per_user_predictions` so the arms can be compared on the accounts they
   share, but nothing in the notebook does that reconciliation yet.

### Planned experiments, not yet run

- **The `posts+stats-long` arm on ModernBERT**: implemented, audited and not
  yet run. It is the only way to tell genuine saturation of the earliness curve
  from the 512-token budget, and the 8-post arm's curve was still climbing at
  its last point.
- **The `posts` control**, three runs of `bert-medium`, without which the
  contribution of the behaviour block cannot be separated from the removal of
  the profile block.
- ~~`full+stats`, the behaviour block on top of the full template~~ - **done**,
  and it adds nothing: +0.0004 macro-F1 over `full` on the same three seeds,
  McNemar p >= 0.68 on each, while bot-F1 drops on every seed and short prefixes
  get slightly worse. See `RESULTS.md` section 7.
- ~~`posts+stats` re-run on the audited template~~ - **done**, and it changed
  nothing: 0.6673 macro-F1 against 0.6674 for the block it replaced, with
  McNemar finding no difference on any of the three seeds (p = 0.24, 0.95,
  0.18). What the audit did fix is the operating point, which used to vary
  wildly between seeds. See `RESULTS.md` section 6.
- **Integrated Gradients** for per-field and per-event-position attribution, as
  in the two final heatmaps of LEGOLAS.
- **A non-neural control on the posts-only text**: TF-IDF word and character
  n-grams into a linear model, minutes on CPU, to say whether the encoder adds
  anything over surface lexical statistics.
- **The winning arm on `roberta-base`**, once the ablation has picked one.

---

## 13. References

1. Pasquadibisceglie, V., Appice, A., Malerba, D., Fiameni, G. *Leveraging a
   large language model to predict hospital admissions of emergency department
   patients.* Expert Systems with Applications 287, 128224. **[LEGOLAS]**
2. Feng, S., Wan, H., Wang, N., Li, J., Luo, M. *TwiBot-20: A Comprehensive
   Twitter Bot Detection Benchmark.* CIKM 2021. **[dataset]**
3. Feng, S. et al. *What Does the Bot Say? Opportunities and Risks of Large
   Language Models in Social Media Bot Detection.* ACL 2024,
   arXiv:2402.00371. **[baselines and metadata verbalization]**
4. Hegselmann, S. et al. *TabLLM: Few-shot Classification of Tabular Data with
   Large Language Models.* AISTATS 2023. **[serialization]**
5. Pasquadibisceglie, V., Appice, A., Castellano, G., Malerba, D. *JARVIS:
   Joining Adversarial Training With Vision Transformers in Next-Activity
   Prediction.* IEEE TSC. **[discretization of numerical views]**
