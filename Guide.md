# Olympic AI Guide

# 1. Foundational Task

## 1.1. Natural Language Processing

### Feature Extraction

**1. Task Description**

Apply a (usually frozen) pretrained encoder $f_\theta: \mathcal{V}^T \to \mathbb{R}^d$ that compresses a token sequence into a dense vector whose geometry (cosine / dot product) encodes semantics. The vector is not the deliverable. It feeds a cheap downstream model (linear head, gradient-boosted trees, kNN, clustering).

**2. Input & Output Signatures**

- Input: token ids `x ∈ ℤ^{B×T}`, attention mask `m ∈ {0,1}^{B×T}`.
- Hidden states: `H ∈ ℝ^{B×T×d}` (optionally every layer: `ℝ^{L×B×T×d}`).
- Output: pooled embedding `z ∈ ℝ^{B×d}`, e.g. masked mean $z_b = \frac{\sum_t m_{bt} h_{bt}}{\sum_t m_{bt}}$, then $\hat z = z / \lVert z \rVert_2$.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Sentence/document embedding → classification/regression:* pooled vector + linear or MLP head (cross-entropy / MSE).
  - *Token-level features:* no pooling; per-token head (see Token Classification).
  - *Retrieval / clustering embeddings:* trained with a contrastive objective (InfoNCE); consumed by cosine kNN or k-means / HDBSCAN.
  - *Feature stacking:* embeddings (often PCA-reduced) concatenated with tabular metadata and fed to gradient-boosted trees.
- **Olympic AI Framing:**
  - Text arrives alongside tabular columns, so the real task is a tabular problem with text features.
  - Fine-tuning is forbidden or infeasible in the time limit, so frozen features plus a clever head is the whole game.
  - Documents far longer than the encoder's context window.
  - Multilingual or code-switched text; test set from a different domain than train.
  - No labels at all: clustering, topic discovery, or anomaly detection.

**4. Baseline Pipeline (Pseudo-code)**

```python
encoder = load_pretrained_encoder(ENCODER_PATH)
encoder.eval()                                          # disable dropout

def embed(texts, batch_size=64, max_len=256):
    feats = []
    for batch in chunks(texts, batch_size):
        ids, mask = tokenize(batch, max_len=max_len, pad=True, truncate=True)
        H = encoder(ids, mask)                          # [B, T, d]
        m = mask[..., None]                             # [B, T, 1]
        z = (H * m).sum(axis=1) / m.sum(axis=1)         # masked mean pooling -> [B, d]
        feats.append(l2_normalize(z))
    return concat(feats)                                # [N, d]

X_train, X_test = embed(train_texts), embed(test_texts)
oof, test_pred = zeros(len(X_train), C), zeros(len(X_test), C)
for tr_idx, va_idx in stratified_kfold(y_train, k=5):
    head = fit_linear_classifier(X_train[tr_idx], y_train[tr_idx], l2_reg=1.0)
    oof[va_idx] = head.predict_proba(X_train[va_idx])
    test_pred += head.predict_proba(X_test) / 5
print(metric(y_train, oof))
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Frozen encoder + mean pooling + linear head under 5-fold CV. This sets a floor in minutes.
  2. Benchmark 2–3 encoders (general encoders vs. dedicated sentence-embedding models) and 3–4 pooling choices (CLS, mean, max, mean of last 4 layers) on out-of-fold (OOF) score.
  3. If labels and time allow, move to end-to-end fine-tuning (see Text Classification); otherwise stack embeddings from several encoders with metadata in a tree model.
- **What Moves the Score:**
  - Model-specific input prefixes / instructions (e.g. `"query: "` vs `"passage: "`).
  - Layer selection: for frozen use, intermediate-to-late layers often transfer better than the last layer.
  - Long text: head+tail truncation, or chunk → embed → average / max over chunks.
  - Concatenating embeddings from diverse encoders; PCA whitening before linear / kNN heads.
  - Tuning the head's regularisation strength on OOF.
- **Common Bugs:**
  - Mean pooling without the mask: padding vectors leak in, so embeddings change with batch composition.
  - Encoder left in training mode: dropout makes features non-deterministic.
  - Fitting the scaler / PCA on all training data before CV → optimistic CV score.
  - Tokenizer from a different checkpoint than the encoder.
  - Silent truncation discarding most of each document.
  - Dot product on unnormalised vectors when cosine similarity was intended.

### Fill-Mask

**1. Task Description**

Masked language modelling (MLM): given a corrupted sequence $\tilde{x}$ in which positions $\mathcal{M}$ are replaced by a mask token, estimate $p_\theta(x_i \mid \tilde{x})$ over the vocabulary for each $i \in \mathcal{M}$, trained with $\mathcal{L} = -\sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid \tilde{x})$. In competitions it appears as (a) direct prediction of missing words, (b) domain-adaptive pre-training (DAPT) on unlabelled text, and (c) cloze-style zero/few-shot classification.

**2. Input & Output Signatures**

- Input: `x̃ ∈ ℤ^{B×T}` containing one or more `MASK_ID` positions.
- Output: logits `ℝ^{B×T×|V|}`; at each masked position a distribution over `|V|` → top-k tokens with probabilities.
- Cloze classification: class score $s_c = \log p_\theta(v_c \mid \tilde{x})$ for a verbaliser token $v_c$ per class → `ℝ^{B×C}`.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Single-token prediction:* softmax over the vocabulary, cross-entropy.
  - *Multi-token / whole-word / span infilling:* iterative decoding, or a span-corruption seq2seq model.
  - *Candidate ranking (multiple-choice cloze):* score each option by (pseudo-)log-likelihood; no new head.
  - *Prompt-based classification:* verbaliser tokens per class; cross-entropy restricted to those tokens.
  - *DAPT:* unsupervised MLM loss on in-domain text before task fine-tuning.
- **Olympic AI Framing:**
  - The answer word splits into several sub-tokens, or is out of vocabulary for a single mask.
  - A small labelled set plus a large unlabelled corpus: an implicit hint to run DAPT.
  - Domain jargon or a low-resource language the base model has barely seen.
  - Closed candidate sets with near-synonym distractors.
  - Exact-match scoring that is sensitive to casing, accents and whitespace.

**4. Baseline Pipeline (Pseudo-code)**

```python
mlm = load_pretrained_mlm(MODEL_PATH); mlm.eval()

def predict_mask(text, k=5):
    ids = tokenize(text)                                # contains MASK_ID
    logits = mlm(ids[None])[0]                          # [T, V]
    pos = where(ids == MASK_ID)
    return [topk(softmax(logits[p]), k) for p in pos]

def rank_candidates(text_with_blank, candidates):
    scores = []
    for cand in candidates:
        cand_ids = tokenize(cand, add_special_tokens=False)
        ids = tokenize(text_with_blank.replace("___", MASK_TOKEN * len(cand_ids)))
        logp = log_softmax(mlm(ids[None])[0], axis=-1)  # [T, V]
        pos = where(ids == MASK_ID)
        scores.append(mean(logp[pos, cand_ids]))        # length-normalised log-likelihood
    return candidates[argmax(scores)]

# Domain-adaptive pre-training on unlabelled text
for batch in unlabelled_loader:
    ids, labels = random_mask(batch, p=0.15, p_mask=0.8, p_random=0.1)   # labels = IGNORE elsewhere
    loss = cross_entropy(mlm(ids), labels, ignore_index=IGNORE)
    loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Determine whether targets are single tokens in the model's vocabulary and whether a candidate list exists. With candidates, frame it as ranking, not generation.
  2. Zero-shot with the pretrained MLM to get a baseline.
  3. Continue MLM training on in-domain text with the *same masking distribution as the test* (whole-word masking if test blanks are whole words).
  4. For multi-token answers, compare iterative left-to-right filling with pseudo-log-likelihood ranking.
- **What Moves the Score:**
  - Matching the training masking scheme to the test masking scheme.
  - Length normalisation when comparing candidates with different numbers of sub-tokens.
  - Ensembling MLMs with different tokenizers by averaging *word-level* log-probabilities.
  - DAPT for 1–3 epochs before downstream fine-tuning.
  - Restricting predictions to a plausible vocabulary (words observed in the corpus).
- **Common Bugs:**
  - Hard-coding `"[MASK]"` when the model uses `"<mask>"`.
  - BPE leading-space tokens: `" word"` and `"word"` tokenise differently, so the candidate's ids differ from those in context.
  - Masking special or padding tokens during DAPT; not setting unmasked labels to the ignore index.
  - Comparing summed log-probabilities across candidates of different lengths (biased towards short ones).
  - Lower-casing input for a cased model.

### Text Classification

**1. Task Description**

Learn $p_\theta(y \mid x)$ mapping a token sequence (or a pair of sequences) to one of $C$ classes, a subset of labels, or an ordinal / continuous score.

**2. Input & Output Signatures**

- Input: `ids ∈ ℤ^{B×T}`, mask; pairs packed as `[CLS] a [SEP] b [SEP]` with segment ids.
- Output: logits `ℝ^{B×C}` → softmax (single-label), element-wise sigmoid (multi-label), or a scalar `ℝ^{B×1}` (regression / ordinal).

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Binary:* 1 logit + binary cross-entropy (BCE); tune the decision threshold.
  - *Multi-class:* $C$ logits + cross-entropy (optionally label smoothing).
  - *Multi-label:* $C$ independent logits + per-label BCE (or focal / asymmetric loss); per-class thresholds.
  - *Ordinal / rating:* regression head + MSE, then optimise cut-points; or cumulative-link ordinal heads.
  - *Pair classification (NLI, paraphrase, stance):* cross-encoder input, same heads.
  - *Hierarchical labels:* one head per level, or a joint softmax constrained by the hierarchy.
- **Olympic AI Framing:**
  - Severe class imbalance / long tail, scored by macro-F1 so the majority-class shortcut fails.
  - Noisy crowd-sourced labels; label-distribution shift between train and test.
  - Documents longer than 512 tokens.
  - Social-media noise (emoji, slang, code-mixing) or a low-resource language.
  - Spurious lexical shortcuts: keywords correlated with labels in train but not in test.
  - Metric quirks: quadratic weighted kappa (QWK) for ordinal labels, log-loss that rewards calibration.

**4. Baseline Pipeline (Pseudo-code)**

```python
for fold, (tr_idx, va_idx) in enumerate(stratified_group_kfold(labels, groups, k=5)):
    encoder = load_pretrained_encoder(MODEL_PATH)
    head = Linear(encoder.hidden_dim, NUM_CLASSES)
    params = encoder.params + head.params
    opt = AdamW(params, lr=2e-5, weight_decay=0.01)
    sched = linear_warmup_then_decay(opt, warmup=0.1 * TOTAL_STEPS, total=TOTAL_STEPS)
    best = -inf

    for epoch in range(EPOCHS):
        for ids, mask, y in loader(train[tr_idx], batch=16, shuffle=True, pad_to_longest=True):
            h = masked_mean(encoder(ids, mask), mask)   # [B, d]  (or the CLS vector)
            logits = head(dropout(h, p=0.1))            # [B, C]
            loss = cross_entropy(logits, y, label_smoothing=0.05)
            loss.backward(); clip_grad_norm(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        score = metric(train.y[va_idx], predict(encoder, head, train[va_idx]))
        if score > best:
            best = score; save(encoder, head, f"fold{fold}.ckpt")

    model = load(f"fold{fold}.ckpt")
    oof[va_idx] = predict(model, train[va_idx])
    test_probs += predict(model, test) / 5

thresholds = optimise_thresholds(oof, train.y, metric)   # for F1 / QWK-style metrics
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Choose the head and loss that match the metric; inspect class balance and token lengths.
  2. Fit a TF-IDF + linear model first (minutes). It gives a floor and exposes shortcut words via its coefficients.
  3. Fine-tune a strong encoder (DeBERTa-v3-class for English, mDeBERTa / XLM-R for multilingual) with 5-fold CV.
  4. Iterate: longer context, pooling variants, layer-wise LR decay, then ensembling and pseudo-labelling.
- **What Moves the Score:**
  - Learning-rate sweep (`1e-5`–`3e-5`) with warmup; 2–5 epochs; layer-wise LR decay (0.8–0.95).
  - Metric-aware post-processing: threshold search for F1, cut-point search for QWK, temperature scaling for log-loss.
  - Head+tail truncation or chunk-and-pool for long documents.
  - Class-weighted or focal loss under imbalance.
  - Adversarial weight perturbation (AWP / FGM): small but consistent gains.
  - Pseudo-labelling high-confidence test predictions; distilling an ensemble into one model.
  - Ensembles weighted by OOF score across different backbones.
- **Common Bugs:**
  - Duplicate or near-duplicate texts across folds (stratify *and* group).
  - Tuning thresholds on the same fold used to report the score; tune on the full OOF instead.
  - Label-index mapping in the submission differs from training (`id2label` order).
  - Softmax used for a multi-label problem (or sigmoid for a single-label one).
  - Pair inputs concatenated as one string without a separator token.
  - fp16 overflow → NaN loss when the learning rate is too high.

### Token Classification

**1. Task Description**

Sequence labelling: assign a tag $y_t \in \mathcal{Y}$ to every token or word $x_t$. Spans are encoded with BIO / BIOES tags and decoded back to `(start, end, type)` triples. The model is either factorised per token, $\prod_t p(y_t \mid x)$, or structured (CRF).

**2. Input & Output Signatures**

- Input: words `w_1..w_N` → sub-word ids `ℤ^{B×T}` plus an alignment `word_ids: t ↦ {0..N−1} ∪ {None}` (or character offsets `t ↦ (c_start, c_end)`).
- Output: logits `ℝ^{B×T×|Y|}` → one tag per word `Y^N` → spans `[(char_start, char_end, type)]`.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Named entity recognition (NER):* BIO tags, per-token softmax or CRF; entity-level F1.
  - *POS tagging / chunking:* per-token softmax; token accuracy.
  - *PII / span detection with character offsets:* per-token softmax plus offset mapping.
  - *Nested NER:* span-enumeration classifier, or one tag layer per nesting level.
  - *Word segmentation* (Vietnamese compounds, Chinese): B/I tagging of syllables or characters.
  - *Pointer-style span extraction:* start/end heads instead of tags.
- **Olympic AI Framing:**
  - Extreme `O`-tag dominance (often >95%) and rare entity types.
  - Ground truth given as character offsets over raw text rather than pre-split tokens.
  - Long documents (essays, contracts) beyond the context window.
  - Inconsistent span boundaries in the annotations.
  - Recall-weighted metrics (e.g. F-beta with β > 1).
  - Unseen entity surface forms at test time.

**4. Baseline Pipeline (Pseudo-code)**

```python
def align_labels(words, word_labels, max_len):
    ids, word_ids = tokenize_words(words, max_len=max_len)   # word_ids[t] = word index or None
    labels, prev = [], None
    for w in word_ids:
        if w is None:     labels.append(IGNORE)              # special / padding tokens
        elif w != prev:   labels.append(word_labels[w])      # first sub-token carries the label
        else:             labels.append(IGNORE)              # remaining sub-tokens ignored
        prev = w
    return ids, labels, word_ids

encoder = load_pretrained_encoder(MODEL_PATH); head = Linear(d, NUM_TAGS)
for ids, mask, labels in loader(train):
    logits = head(encoder(ids, mask))                        # [B, T, NUM_TAGS]
    loss = cross_entropy(logits.reshape(-1, NUM_TAGS), labels.reshape(-1), ignore_index=IGNORE)
    loss.backward(); opt.step(); sched.step(); opt.zero_grad()

def decode(logits, word_ids):
    word_tag = {}
    for t, w in enumerate(word_ids):
        if w is not None and w not in word_tag:
            word_tag[w] = argmax(logits[t])
    tags = repair_bio([word_tag[i] for i in sorted(word_tag)])   # I-X after O / B-Y  ->  B-X
    return tags_to_spans(tags)
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Establish the annotation unit (characters vs words) and the metric (entity- vs token-level).
  2. Baseline: encoder + linear token head, first-sub-token labelling, sliding windows over long texts.
  3. Add constrained decoding (CRF or Viterbi with BIO transition constraints), larger encoders, then ensembling by averaging *word-level* probabilities.
  4. Trade precision against recall with a threshold on the `O` probability.
- **What Moves the Score:**
  - `O`-threshold: predict an entity when $P(O) < \tau$, with τ tuned on OOF. This is crucial for recall-weighted metrics.
  - Sliding windows with stride; for each token, keep predictions from the window where it is most central.
  - Rule-based post-processing: regexes for structured entities (emails, phone numbers, IDs), gazetteers of known names.
  - Down-sampling windows that contain no entities.
  - Character-level probability averaging when ensembling models with different tokenizers.
- **Common Bugs:**
  - Labels shifted by one position because special tokens were not accounted for.
  - A slow tokenizer without word ids / offset mapping.
  - Inclusive vs exclusive span ends (off-by-one character offsets).
  - Reporting token accuracy when the competition scores entity-level F1.
  - BPE leading-space tokens making offsets include the preceding space.
  - Truncation silently dropping entities at the end of long documents.

### Sentence Similarity

**1. Task Description**

Learn a scoring function $s(a, b) \in \mathbb{R}$ for semantic equivalence or relatedness between two texts. Two parametrisations: **bi-encoder** $s = \cos(f(a), f(b))$ (independent encoding, indexable) and **cross-encoder** $s = g([a; b])$ (joint encoding, more accurate, not indexable).

**2. Input & Output Signatures**

- Input: pair `(a, b)` of token sequences.
- Bi-encoder intermediate: `f(a), f(b) ∈ ℝ^{d}`.
- Output: scalar `ℝ` (e.g. 0–5 similarity), probability `[0, 1]` (paraphrase), or 3-way logits `ℝ^{3}` (entailment / neutral / contradiction).

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Semantic textual similarity (STS):* regression head; MSE / cosine-similarity loss; Pearson or Spearman.
  - *Paraphrase / duplicate detection:* binary head, BCE.
  - *Natural language inference:* 3-way softmax.
  - *Semantic search / mining:* bi-encoder with in-batch-negative InfoNCE (multiple-negatives ranking loss).
  - *Cross-lingual similarity:* multilingual encoders, the same losses.
- **Olympic AI Framing:**
  - Asymmetric pairs: short phrase vs long passage.
  - Domain-specific phrases where an extra context field changes the meaning (e.g. a category code given per pair).
  - Discrete, noisy ordinal scores.
  - Rank-based metric (Spearman): calibration is irrelevant, only ordering matters.
  - Lexical-overlap traps: high word overlap but different meaning.

**4. Baseline Pipeline (Pseudo-code)**

```python
# Option A: bi-encoder, zero-shot
emb = load_pretrained_encoder(EMB_PATH)
za = l2_normalize(masked_mean(emb(tokenize(A))))        # [N, d]
zb = l2_normalize(masked_mean(emb(tokenize(B))))        # [N, d]
sim = (za * zb).sum(-1)                                  # [N] cosine similarity

# Option B: cross-encoder regression, fine-tuned
encoder = load_pretrained_encoder(MODEL_PATH); head = Linear(d, 1)
for a, b, y in loader(train):                            # y rescaled to [0, 1]
    ids, mask, seg = tokenize_pair(a, b, context=CTX)    # optional extra context field
    pred = sigmoid(head(encoder(ids, mask, seg)[:, 0])).squeeze(-1)
    loss = binary_cross_entropy(pred, y)                 # soft targets; or MSE
    loss.backward(); opt.step(); sched.step(); opt.zero_grad()

def score(a, b):                                         # symmetric test-time augmentation
    return 0.5 * (cross_encoder(a, b) + cross_encoder(b, a))
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. If the test set is a fixed list of pairs, the cross-encoder is the main model; if it requires search over a corpus, the bi-encoder is.
  2. Zero-shot bi-encoder cosine gives a baseline Spearman score immediately.
  3. Fine-tune a cross-encoder with 5-fold CV *grouped by anchor text*.
  4. Inject side information into the input, then ensemble several backbones with rank averaging.
- **What Moves the Score:**
  - Swap augmentation `(a, b) → (b, a)` in training and as test-time augmentation.
  - BCE with soft targets often beats MSE for bounded scores.
  - Rank-averaging ensemble members when the metric is Spearman.
  - Adding context fields (category description, domain) to the input text.
  - For contrastive training: large batches, hard negatives mined with the current bi-encoder, deduplicated batches.
- **Common Bugs:**
  - The same anchor sentence appearing in train and validation folds.
  - In-batch false negatives (duplicate texts treated as negatives).
  - Forgetting L2 normalisation, so the dot product ≠ cosine.
  - Optimising for Pearson when the metric is Spearman (or vice versa).
  - Rescaling labels for training but not inverting the scaling for the submission.

### Text Ranking

**1. Task Description**

Given a query $q$ and corpus $\mathcal{D}$, output a permutation of (a subset of) $\mathcal{D}$ maximising a ranking metric. Standard design is two-stage: a cheap retriever selects $R(q) \subset \mathcal{D}$, $|R| = k \ll |\mathcal{D}|$, and an expensive reranker $s(q, d)$ reorders $R(q)$.

**2. Input & Output Signatures**

- Input: query string; corpus of `N` documents (often chunked into passages).
- Index: sparse term weights, or dense passage embeddings `ℝ^{N_p×d}`.
- Output: ranked doc ids `[d_1, …, d_k]` with scores → `ℝ^{Q×k}`.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Sparse retrieval* (BM25): no training.
  - *Dense retrieval:* bi-encoder, InfoNCE with in-batch and hard negatives.
  - *Reranking:* pointwise (BCE on relevance), pairwise (margin / RankNet), listwise (softmax cross-entropy over a candidate list, LambdaRank).
  - *Graded relevance:* targets in {0, 1, 2, 3}, optimised for nDCG.
  - *Feature-based learning-to-rank:* hand-crafted features + gradient-boosted LambdaMART.
- **Olympic AI Framing:**
  - Large corpus under a strict time / memory budget.
  - Few labelled queries, mostly unlabelled documents.
  - Vocabulary mismatch between queries and documents; cross-lingual retrieval.
  - Incomplete relevance labels: unlabelled relevant documents are counted as negatives.
  - Metric at a fixed cut-off (MAP@25, Recall@50).

**4. Baseline Pipeline (Pseudo-code)**

```python
passages, owner = chunk_documents(corpus, size=256, overlap=64)   # owner[p] = doc id

# Stage 1: hybrid candidate generation with reciprocal rank fusion (RRF)
sparse = build_bm25_index(passages)
P = l2_normalize(embed(passages))                       # [Np, d]
def retrieve(q, k=200):
    r_sparse = rank_positions(sparse.score(q))          # passage -> rank (0 = best)
    r_dense  = rank_positions(P @ l2_normalize(embed([q]))[0])
    cands = union(top(r_sparse, k), top(r_dense, k))
    fused = {p: 1 / (60 + r_sparse[p]) + 1 / (60 + r_dense[p]) for p in cands}
    return top_k_by_value(fused, k)

# Stage 2: cross-encoder reranking, document score = best passage score
def rerank(q, cands, k_out=10):
    s = {p: cross_encoder(tokenize_pair(q, passages[p])) for p in cands}
    doc_score = {}
    for p, v in s.items():
        doc_score[owner[p]] = max(doc_score.get(owner[p], -inf), v)
    return sorted(doc_score, key=doc_score.get, reverse=True)[:k_out]

# Reranker training, listwise with hard negatives from stage 1
for q, pos, hard_negs in loader(train_queries):
    logits = stack([cross_encoder(q, d) for d in [pos] + hard_negs])   # [1 + n_neg]
    loss = cross_entropy(logits[None], target=[0])
    loss.backward(); opt.step(); opt.zero_grad()
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Measure stage-1 Recall@k on its own: it is the ceiling for everything after.
  2. BM25 baseline → add zero-shot dense retrieval → fuse → add a zero-shot reranker.
  3. Fine-tune the reranker (then the retriever) with hard negatives; tune `k` to balance recall against reranking time.
- **What Moves the Score:**
  - Hybrid sparse + dense retrieval with RRF.
  - Hard-negative mining, filtering likely false negatives by score margin.
  - Document expansion (generated questions per document) or query expansion.
  - Prepending titles to passages; max-over-passages document scoring.
  - Listwise losses over pointwise ones; ensembling rerankers.
- **Common Bugs:**
  - Splitting validation by (query, doc) pairs instead of by query → the same query in train and validation.
  - Returning the same document several times after chunking (deduplicate by owner).
  - Different tokenisation / stemming for queries and documents in BM25.
  - Missing the query / passage prefixes an embedding model requires.
  - nDCG computed with the ideal DCG over retrieved items only instead of all relevant items.
  - Submitting fewer than `k` results or in the wrong order.

### Question Answering

**1. Task Description**

Extractive QA: given question $q$ and context $c = (c_1, \dots, c_n)$, predict the span $(s, e)$ maximising $p(s)\,p(e)$ subject to $s \le e$ and $e - s < L_{\max}$, or output "no answer". Variants include multiple choice, open-domain (retriever + reader) and generative QA.

**2. Input & Output Signatures**

- Input: `[CLS] q [SEP] c_window [SEP]` → `ℤ^{B×T}`, plus offsets `t ↦ (char_start, char_end)` (None for question / special tokens).
- Output: start and end logits `ℝ^{B×T}` each → character span `(a, b)` → answer `c[a:b]`, or the empty string.
- Multiple choice: `ℝ^{B×K}` over K options.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Extractive span QA:* start/end heads, cross-entropy on each.
  - *Unanswerable questions:* null score from the CLS position, compared with the best span via a tuned margin.
  - *Multiple choice:* encode each (context, question, option) → scalar; softmax over options.
  - *Open-domain QA:* retriever (see Text Ranking) + reader.
  - *Generative / abstractive QA:* seq2seq or decoder LM, token cross-entropy.
  - *Yes/No + span:* extra classification head alongside the span head.
- **Olympic AI Framing:**
  - Long contexts (full articles) requiring sliding windows.
  - Low-resource languages and scripts.
  - Noisy answer offsets; the answer string occurs several times in the context.
  - A sizeable share of unanswerable questions.
  - Word-level Jaccard / F1, which punishes stray punctuation and partial words.
  - Distractor paragraphs.

**4. Baseline Pipeline (Pseudo-code)**

```python
def featurize(q, ctx, max_len=384, stride=128):
    # overlapping windows; only the context is truncated
    return tokenize_pair(q, ctx, max_len=max_len, stride=stride,
                         truncate="context_only", return_offsets=True)

encoder = load_pretrained_encoder(MODEL_PATH); head = Linear(d, 2)
# targets: token indices covering the answer's char span, or (0, 0) = CLS if not inside this window
for ids, mask, start_t, end_t in loader(train_windows):
    start_logits, end_logits = split(head(encoder(ids, mask)), 2, axis=-1)   # [B, T] each
    loss = 0.5 * (cross_entropy(start_logits, start_t) + cross_entropy(end_logits, end_t))
    loss.backward(); opt.step(); sched.step(); opt.zero_grad()

def postprocess(windows, all_start, all_end, ctx, n_best=20, max_ans_len=30, null_margin=0.0):
    best, best_score = "", -inf
    for w, sl, el in zip(windows, all_start, all_end):
        for s in topk_indices(sl, n_best):
            for e in topk_indices(el, n_best):
                if w.offsets[s] is None or w.offsets[e] is None: continue
                if e < s or e - s + 1 > max_ans_len:             continue
                if sl[s] + el[e] > best_score:
                    best_score = sl[s] + el[e]
                    best = ctx[w.offsets[s][0] : w.offsets[e][1]]
    null_score = min(sl[0] + el[0] for sl, el in zip(all_start, all_end))
    return "" if null_score - best_score > null_margin else best.strip()
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Read the submission format: extractive, generative or multiple choice.
  2. Zero-shot with a checkpoint already fine-tuned on a large extractive QA dataset gives the baseline.
  3. Fine-tune on competition data with folds *grouped by context*.
  4. Scale to larger (multilingual) models; add external QA data if the rules allow; ensemble at the character level.
- **What Moves the Score:**
  - Two-stage fine-tuning: large public QA data, then competition data.
  - Character-level start/end probability averaging across models with different tokenizers.
  - Post-processing: strip punctuation, snap partial words to word boundaries.
  - Tuning `max_len` / `stride`; down-sampling windows without an answer.
  - Tuning the null margin on OOF.
- **Common Bugs:**
  - Offsets not masked for question tokens, so the "answer" is taken from the question.
  - Wrong `answer_start` in the data. Fix it by matching the answer text to its nearest occurrence.
  - Windows that contain only part of the answer get inconsistent labels.
  - Folds grouped by question instead of by context → leakage.
  - Independent argmax of start and end, producing `e < s`.
  - Unicode normalisation (combining marks, Indic scripts) shifting character offsets.

### Text Generation

**1. Task Description**

Model $p_\theta(y \mid x) = \prod_t p_\theta(y_t \mid y_{<t}, x)$, trained with teacher-forced cross-entropy and used through a decoding algorithm $\hat{y} = \mathrm{decode}(p_\theta, x)$. Covers summarisation, translation, data-to-text, instruction following and structured output.

**2. Input & Output Signatures**

- Input: source / prompt ids `ℤ^{B×T_in}`.
- Training: logits `ℝ^{B×T×|V|}`. Decoder-only: sequence `[x; y]` with loss only on `y`. Encoder-decoder: `encoder(x)`, `decoder(y_{<t})`.
- Output: generated ids `ℤ^{B×T_out}` → strings (optionally parsed into JSON, labels, etc.).

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - Summarisation (ROUGE), translation (BLEU / chrF), paraphrasing, data-to-text, code generation (pass@k), dialogue, structured output (schema-validated JSON).
  - *Classification-as-generation:* LM head restricted to verbaliser tokens.
  - *Training regimes:* prompting only (zero / few-shot), full fine-tuning, parameter-efficient fine-tuning (LoRA), preference optimisation (rare in contests).
- **Olympic AI Framing:**
  - Tiny labelled set + a large provided LLM + limited GPU → prompting or LoRA, not full fine-tuning.
  - Outputs checked by a strict parser; malformed output scores zero.
  - Low-resource language pairs.
  - Length-sensitive metrics (ROUGE recall vs F-measure).
  - Inference time limit: model size and decoding strategy must fit the test set.

**4. Baseline Pipeline (Pseudo-code)**

```python
model = load_pretrained_causal_lm(MODEL_PATH, dtype="fp16")
model = add_low_rank_adapters(model, rank=16, alpha=32, targets="attention_projections")  # optional

def build_example(src, tgt):
    p = tokenize(PROMPT_TEMPLATE.format(src=src))
    a = tokenize(tgt) + [EOS_ID]                        # the model must learn to stop
    return p + a, [IGNORE] * len(p) + a                 # loss only on the answer tokens

for ids, labels in loader(train, pad_side="right"):
    logits = model(ids)                                 # [B, T, V]
    loss = cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=IGNORE)   # next-token shift
    loss.backward(); opt.step(); sched.step(); opt.zero_grad()

def generate(srcs, max_new=128):
    ids, mask = tokenize_batch([PROMPT_TEMPLATE.format(src=s) for s in srcs], pad_side="left")
    out = model.generate(ids, mask, max_new_tokens=max_new, num_beams=4,
                         no_repeat_ngram_size=3, eos_token_id=EOS_ID)
    return detokenize(out[:, ids.shape[1]:])            # strip the prompt
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Read the metric and output constraints; measure a zero-shot prompt baseline *and* time per example on a validation subset.
  2. Few-shot prompting with retrieved similar training examples.
  3. LoRA fine-tuning once a few hundred labelled pairs exist.
  4. Tune decoding, then add candidate reranking (minimum Bayes risk, a scoring model) and constrained decoding.
- **What Moves the Score:**
  - Decoding hyperparameters (beams, `length_penalty`, max length) tuned on the metric. ROUGE / BLEU often swing by several points.
  - Minimum Bayes risk (MBR) decoding: sample N candidates, pick the one with highest average metric against the others.
  - Retrieval-selected few-shot examples.
  - Output validation and repair (parse → fix → fall back to the next candidate); truncate at stop sequences.
  - Loss masking on the prompt; sequence packing for throughput.
- **Common Bugs:**
  - Right padding during batched generation with a decoder-only model → corrupted outputs.
  - Not stripping the prompt from the generated ids.
  - No EOS in training targets, so the model never stops.
  - `pad = eos` with pad positions ignored in the loss, which also ignores the real EOS.
  - Prompt format that does not match the instruct model's chat template.
  - Tokenising prompt and answer separately vs jointly changes the boundary tokens.
  - OOM from `max_new_tokens × num_beams × batch`.

## 1.2. Computer Vision

### Image Feature Extraction

**1. Task Description**

Map an image $I \in \mathbb{R}^{3 \times H \times W}$ to a global descriptor $z = f_\theta(I) \in \mathbb{R}^d$, or to dense patch features $F \in \mathbb{R}^{d \times h \times w}$, using a pretrained backbone. Downstream heads (linear probe, kNN, tree models, decoders) consume these features.

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]` float tensor, normalised with the backbone's own mean/std, at the backbone's native resolution (ViTs: multiple of the patch size, e.g. 14 or 16).
- Output:
  - Global: `[B, d]` (CLS token or global average pooling).
  - Patch tokens (ViT): `[B, N, d]` with $N = (H/p)(W/p)$.
  - CNN feature maps: `[B, d, H/32, W/32]` (or a multi-scale pyramid).

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Global embedding → classification:* linear probe (cross-entropy) or kNN.
  - *Dense features → segmentation / correspondence:* per-patch heads or matching.
  - *Retrieval / deduplication:* cosine similarity; optionally metric-learning fine-tuning.
  - *Clustering / novelty detection:* k-means, density estimation, distance-to-train.
  - *Multi-scale features:* feature pyramid feeding detection heads.
- **Olympic AI Framing:**
  - Large domain gap from natural-image pretraining (medical, satellite, microscopy, documents).
  - Few-shot: a handful of labelled images per class.
  - Unlabelled image sets (clustering / anomaly detection).
  - Frozen backbone imposed by rules or compute.
  - Mixed resolutions and aspect ratios; tiny discriminative regions.

**4. Baseline Pipeline (Pseudo-code)**

```python
backbone = load_pretrained_vision_backbone(PATH); backbone.eval()
MEAN, STD, SIZE = backbone.norm_mean, backbone.norm_std, backbone.input_size

def preprocess(img):
    img = apply_exif_orientation(to_rgb(img))
    img = center_crop(resize_shorter_side(img, SIZE), SIZE)
    x = to_float_tensor(img) / 255.0                    # [3, H, W]
    return (x - MEAN[:, None, None]) / STD[:, None, None]

def pool(tokens):                                       # tokens: [B, 1 + N, d]
    return concat([tokens[:, 0], tokens[:, 1:].mean(1)], axis=-1)   # CLS ‖ mean patch -> [B, 2d]

def extract(images, bs=128, flip_tta=True):
    feats = []
    for batch in chunks(images, bs):
        x = stack([preprocess(im) for im in batch])     # [B, 3, H, W]
        z = pool(backbone(x))
        if flip_tta:
            z = 0.5 * (z + pool(backbone(flip_horizontal(x))))
        feats.append(l2_normalize(z))
    return concat(feats)                                # [N, 2d]
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Benchmark 2–3 backbone families: self-supervised (DINOv2), language-supervised (CLIP / SigLIP), supervised CNN (ConvNeXt / EfficientNet). Use a linear probe and kNN under CV.
  2. Choose pooling and input resolution on OOF score.
  3. Decide on frozen vs fine-tuned from the gap between the linear probe and a short fine-tuning run.
- **What Moves the Score:**
  - Higher input resolution (interpolating ViT position embeddings) for small details.
  - Concatenating features across backbones and across intermediate layers.
  - Flip / multi-crop test-time augmentation (TTA) on embeddings.
  - PCA whitening; standardising features for linear heads.
- **Common Bugs:**
  - Wrong normalisation statistics (ImageNet vs CLIP mean/std differ).
  - BGR channel order from OpenCV fed to an RGB model.
  - Grayscale / RGBA / palette images not converted to 3-channel RGB.
  - EXIF orientation ignored, so phone photos arrive rotated.
  - BatchNorm left in training mode during extraction.
  - `uint8` tensors not cast to float before normalisation.

### Image Classification

**1. Task Description**

Learn $p_\theta(y \mid I)$ with $y \in \{1, \dots, C\}$ (single-label), $y \in \{0,1\}^C$ (multi-label), or $y \in \mathbb{R}$ / ordinal grades.

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]`.
- Output: logits `[B, C]` → softmax (single-label) / sigmoid (multi-label); `[B, 1]` for regression / ordinal grading.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Binary / multi-class:* cross-entropy, label smoothing.
  - *Multi-label:* per-class BCE or asymmetric loss; per-class thresholds.
  - *Fine-grained:* subtle inter-class differences; higher resolution, metric-learning heads (ArcFace).
  - *Ordinal grading* (disease severity, quality scores): regression + optimised cut-points; QWK metric.
  - *Open-set:* reject "unknown" via a confidence or distance threshold.
- **Olympic AI Framing:**
  - Long-tailed class distribution; rare classes with a few examples.
  - Distribution shift: test images from a different device, site, season or style.
  - Noisy labels.
  - Gigapixel / high-resolution inputs (pathology, satellite) that require tiling and aggregation.
  - Near-duplicate images (same patient / object / scene) across splits.
  - Tabular metadata alongside images.
  - An "other / unknown" class at test time.

**4. Baseline Pipeline (Pseudo-code)**

```python
train_tf = compose(random_resized_crop(SIZE, scale=(0.6, 1.0)), hflip(p=0.5),
                   color_jitter(0.2), to_tensor(), normalize(MEAN, STD))
eval_tf  = compose(resize(SIZE), center_crop(SIZE), to_tensor(), normalize(MEAN, STD))

for fold, (tr, va) in enumerate(stratified_group_kfold(labels, groups, k=5)):
    model = load_pretrained_backbone(PATH)
    model.head = Linear(model.dim, C)                   # new head
    opt = AdamW([{"params": model.backbone_params, "lr": 1e-4},
                 {"params": model.head_params,     "lr": 1e-3}], weight_decay=0.05)
    sched = cosine_with_warmup(opt, warmup_epochs=1, total_epochs=EPOCHS)
    ema = exponential_moving_average(model, decay=0.999)

    for epoch in range(EPOCHS):
        for x, y in loader(train[tr], transform=train_tf, shuffle=True):
            x, y_a, y_b, lam = mixup_or_cutmix(x, y, alpha=0.2)
            logits = model(x)
            loss = lam * cross_entropy(logits, y_a) + (1 - lam) * cross_entropy(logits, y_b)
            backward_mixed_precision(loss); opt.step(); sched.step(); opt.zero_grad()
            ema.update(model)
        evaluate_and_checkpoint(ema.model, train[va], eval_tf, metric, f"fold{fold}.ckpt")

def predict_tta(model, x):
    return 0.5 * (softmax(model(x)) + softmax(model(hflip(x))))
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. EDA: class counts, resolution histogram, duplicate search (embedding similarity), and a grid of samples per class.
  2. Linear probe on frozen features → fine-tune a mid-size CNN / ViT at near-native resolution with 5-fold CV.
  3. Read the confusion matrix and the misclassified images, then target the failure mode (resolution, augmentation, imbalance).
  4. Scale: resolution, architecture diversity, longer schedules, pseudo-labels, ensemble.
- **What Moves the Score:**
  - Input resolution: often the largest single gain for fine-grained or small-object problems.
  - MixUp / CutMix, RandAugment, label smoothing.
  - Exponential moving average (EMA) of weights.
  - Discriminative learning rates; gradual unfreezing.
  - TTA (flip, multi-scale).
  - Long-tail handling: class-balanced sampling, logit adjustment, focal loss.
  - Pseudo-labelling; distillation from an ensemble.
  - Fusing image embeddings with metadata.
- **Common Bugs:**
  - Label-changing augmentation (flipping text, digits, chirality, left/right anatomy).
  - Random augmentation left in the validation / test transform.
  - Leakage of near-duplicates or same-subject images across folds.
  - Class-index mapping mismatch (sorted folder names vs CSV order).
  - BatchNorm with tiny batches (freeze BN statistics or use GroupNorm).
  - Submitting logits instead of probabilities, or columns in the wrong order.

### Object Detection

**1. Task Description**

Predict a set $\{(b_i, c_i, s_i)\}_{i=1}^{K}$ of bounding boxes $b_i \in \mathbb{R}^4$, class labels $c_i$ and confidences $s_i$ covering every object instance in an image.

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]` (resized / letterboxed).
- Targets per image (variable count): `boxes [n_i, 4]`, `labels [n_i]`.
- Output per image after post-processing: `[K, 6] = [x_min, y_min, x_max, y_max, score, class]` in *original* image pixel coordinates.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *One-stage, anchor-based or anchor-free* (YOLO family, FCOS, RetinaNet): classification (focal / BCE) + box regression (GIoU / CIoU / distribution focal loss) + objectness; requires NMS.
  - *Two-stage* (Faster R-CNN): region proposal network + RoI classification / regression heads.
  - *Set prediction* (DETR family): Hungarian matching between predictions and targets; no NMS.
  - *Oriented boxes* (aerial imagery): an extra angle parameter.
  - *Small / dense objects; counting-by-detection.*
- **Olympic AI Framing:**
  - Tiny objects in very large images → tiling.
  - Many images with no objects; class imbalance across categories.
  - Missing or inconsistent boxes in the annotations.
  - Unusual box encodings in the CSV (normalised, centre-based, width/height).
  - Metric details: mAP@0.5 vs mAP@[0.5:0.95], or F-beta at a set of IoU thresholds.
  - Sensor or lighting shift between train and test.

**4. Baseline Pipeline (Pseudo-code)**

```python
def to_xyxy(box, fmt, W, H):
    if fmt == "coco":                                   # x_min, y_min, w, h (pixels)
        x, y, w, h = box; return [x, y, x + w, y + h]
    if fmt == "yolo":                                   # cx, cy, w, h (normalised 0-1)
        cx, cy, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return box                                          # already x_min, y_min, x_max, y_max

def iou(a, b):                                          # both xyxy
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (area(a) + area(b) - inter + 1e-9)

detector = load_pretrained_detector(PATH, num_classes=C)
for imgs, targets in loader(train, transform=geometric_aug_with_boxes):   # boxes move with the image
    losses = detector(imgs, targets)                    # {cls, box, obj, ...}
    sum(losses.values()).backward(); opt.step(); opt.zero_grad()

def predict(img, conf_th=0.001, iou_th=0.6):
    x, scale, (pad_x, pad_y) = letterbox(img, SIZE)
    boxes, scores, cls = decode(detector(x))            # model-space xyxy
    m = scores > conf_th
    boxes, scores, cls = boxes[m], scores[m], cls[m]
    k = class_wise_nms(boxes, scores, cls, iou_th)
    boxes, scores, cls = boxes[k], scores[k], cls[k]
    boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / scale              # undo letterbox
    return clip(boxes, 0, [img.W, img.H, img.W, img.H]), scores, cls
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Draw ground-truth boxes on ~20 images to confirm the format; plot the distribution of box sizes relative to the image → choose input resolution or tiling.
  2. Baseline: pretrained one-stage detector, default hyperparameters, moderate resolution.
  3. Iterate on resolution / tiling, augmentation (mosaic, scale jitter), epochs, model size.
  4. Ensemble models and TTA with weighted boxes fusion.
- **What Moves the Score:**
  - Weighted boxes fusion (WBF) for ensembles / TTA: usually better than NMS for mAP.
  - Larger input size; overlapping tiles merged with NMS / WBF.
  - Mosaic, MixUp, copy-paste augmentation.
  - Very low confidence threshold for mAP; tuned threshold for F-score metrics.
  - Re-clustered anchors for unusual aspect ratios (anchor-based models).
  - A controlled share of empty images as negatives.
- **Common Bugs:**
  - Box format confusion: xywh vs xyxy, pixels vs normalised, centre vs top-left.
  - Forgetting to undo the resize / letterbox at inference.
  - `(H, W)` vs `(W, H)` swapped in target sizes.
  - Class ids off by one (background reserved at 0 in some frameworks; 1-indexed annotation files).
  - Geometric augmentation applied to images but not to boxes; zero-area boxes after cropping not removed.
  - Degenerate boxes (`x_max < x_min`) producing NaN losses.
  - Pixel convention off-by-one (inclusive coordinates with a `+1` in area).

### Image Segmentation

**1. Task Description**

Dense prediction. **Semantic:** $Y \in \{0, \dots, C-1\}^{H \times W}$. **Instance:** a set of (binary mask, class, score) triples, one per object. **Panoptic:** a unified map of "stuff" classes and "thing" instances.

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]`.
- Semantic: logits `[B, C, H, W]` → argmax → integer mask `[B, H, W]`; binary: `[B, 1, H, W]` + sigmoid.
- Instance: `K × ([H, W] bool mask, class, score)`.
- Submission: often run-length encoding (RLE) strings.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Semantic:* cross-entropy + Dice / Lovász; mean IoU.
  - *Binary:* BCE + Dice; Dice / IoU.
  - *Instance:* Mask R-CNN-style or query-based (Mask2Former); mask AP.
  - *Panoptic:* panoptic quality (PQ).
  - *Promptable:* points / boxes → masks (SAM-style), useful for pseudo-labelling.
  - *3D volumetric* (CT / MRI): 2.5D or 3D U-Net.
- **Olympic AI Framing:**
  - Extreme foreground / background imbalance (tiny lesions, cracks).
  - Many images with empty masks → an image-level "is there anything?" gate.
  - Very large images → tiling with overlap.
  - Thin structures (vessels, roads) where boundary accuracy dominates.
  - Coarse or noisy annotations.
  - RLE conventions (column-major, 1-indexed starts).

**4. Baseline Pipeline (Pseudo-code)**

```python
model = encoder_decoder(backbone=load_pretrained_backbone(PATH), num_classes=C)   # U-Net / FPN style

def seg_loss(logits, mask, smooth=1.0):                 # logits [B, C, H, W], mask [B, H, W]
    ce = cross_entropy(logits, mask, ignore_index=IGNORE)
    valid = (mask != IGNORE)[:, None]                   # [B, 1, H, W]
    p = softmax(logits, axis=1) * valid
    oh = one_hot(where(mask == IGNORE, 0, mask), C).permute(0, 3, 1, 2) * valid
    inter = (p * oh).sum((2, 3)); denom = p.sum((2, 3)) + oh.sum((2, 3))
    dice = 1 - ((2 * inter + smooth) / (denom + smooth)).mean()
    return ce + dice

for img, mask in loader(train, transform=joint_aug):    # same geometry for both; mask uses nearest
    loss = seg_loss(model(img), mask)
    loss.backward(); opt.step(); opt.zero_grad()

def predict_large(img, tile=512, overlap=128):
    prob, weight = zeros(C, H, W), zeros(H, W)
    g = gaussian_window(tile)
    for y0, x0 in sliding_windows(H, W, tile, overlap):
        crop = img[:, y0:y0 + tile, x0:x0 + tile]
        p = 0.5 * (softmax(model(crop), 0) + hflip(softmax(model(hflip(crop)), 0)))   # flip TTA
        prob[:, y0:y0 + tile, x0:x0 + tile] += p * g
        weight[y0:y0 + tile, x0:x0 + tile] += g
    return argmax(prob / weight, axis=0)                # upsample/blend probabilities, argmax last
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Identify semantic vs instance; compute mask statistics (coverage per image, % empty, object size distribution).
  2. Baseline: U-Net / FPN with a pretrained encoder, CE (or BCE) + Dice, moderate resolution.
  3. Iterate on encoder strength, resolution / tiling and loss mix, then post-processing (threshold, minimum component size).
  4. Ensemble probability maps and add TTA.
- **What Moves the Score:**
  - Loss aligned with the metric: Dice / Lovász / focal combinations for IoU / Dice.
  - Probability threshold and minimum connected-component area tuned on OOF.
  - Two-stage: a classifier predicts empty vs non-empty; the segmenter runs only on non-empty images.
  - Flip / rotate TTA (where the domain allows), multi-scale inference.
  - Overlapping tiles with Gaussian blending.
  - Deep supervision; EMA weights.
- **Common Bugs:**
  - Resizing masks with bilinear interpolation → fractional / invalid class ids.
  - Image and mask augmentations not synchronised.
  - RLE ordering errors (row- vs column-major, 0- vs 1-indexed).
  - Argmax before upsampling (blocky edges).
  - Dice computed per batch when the metric is per image; empty-vs-empty masks scored as 1 or 0 differently from the official metric.
  - Unlabelled pixel value (e.g. 255) not set as the ignore index.

### Image Retrieval

**1. Task Description**

Given a query image $q$ and a gallery $\mathcal{G}$, rank gallery items by $s(q, g) = \langle f(q), f(g) \rangle$, where $f$ is trained so that images of the same instance / identity lie close together (metric learning). Covers instance retrieval (landmarks, products), re-identification and near-duplicate detection.

**2. Input & Output Signatures**

- Input: query images `[M, 3, H, W]`; gallery images `[N, 3, H, W]` (pre-embedded to `[N, d]`).
- Output: top-k gallery ids with scores `[M, k]`; in open-set settings, a special "new / unknown" id inserted where scores fall below a threshold.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Instance-level retrieval:* the same physical object or place.
  - *Category-level retrieval:* the same class.
  - *Re-identification* (people, animals, vehicles): across cameras or time.
  - *Near-duplicate / copy detection:* robust to edits and augmentations.
  - *Cross-domain retrieval* (sketch → photo, user photo → catalogue).
  - *Losses:* margin-based softmax (ArcFace / CosFace), triplet loss with hard mining, contrastive / InfoNCE.
- **Olympic AI Framing:**
  - Thousands of identities with 1–5 images each; test identities unseen in training (open-set).
  - A "new individual" answer that needs a calibrated threshold.
  - Distractor-heavy galleries.
  - Query / gallery domain shift.
  - MAP@k with a single correct answer per query.

**4. Baseline Pipeline (Pseudo-code)**

```python
class ArcFaceHead:                                      # additive angular margin softmax
    def __init__(self, d, n_classes, s=30.0, m=0.3):
        self.W = init_normal([n_classes, d]); self.s, self.m = s, m
    def loss(self, z, y):
        cos_t = l2_normalize(z) @ l2_normalize(self.W).T          # [B, n_classes]
        theta = arccos(clip(cos_t, -1 + 1e-7, 1 - 1e-7))
        logits = where(one_hot(y, cos_t.shape[1]), cos(theta + self.m), cos_t) * self.s
        return cross_entropy(logits, y)

backbone = load_pretrained_backbone(PATH)
neck = Sequential(GeMPooling(p=3), Linear(backbone.dim, 512), BatchNorm(512))
head = ArcFaceHead(512, N_IDENTITIES)
for x, y in loader(train):
    loss = head.loss(neck(backbone(x)), y)
    loss.backward(); opt.step(); opt.zero_grad()

embed = lambda imgs: l2_normalize(neck(backbone(imgs)))
G, Q = embed(gallery), embed(queries)                   # [N, 512], [M, 512]
S = Q @ G.T                                             # [M, N] cosine (compute in chunks if large)
# Query expansion: refine each query with its top-3 neighbours, then re-score
Q2 = l2_normalize(Q + G[argsort(-S, axis=1)[:, :3]].mean(axis=1))
ranked = argsort(-(Q2 @ G.T), axis=1)[:, :K]
# Open-set: insert NEW_ID at the first rank whose score < tau (tau tuned on validation)
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Zero-shot: strong pretrained embeddings (DINOv2 / CLIP) + cosine kNN → baseline mAP.
  2. Fine-tune with ArcFace on identity labels, validating on *held-out identities* to mimic the test.
  3. Add GeM pooling, higher resolution, re-ranking and threshold tuning for the "new" class.
  4. Ensemble by concatenating L2-normalised embeddings from several models.
- **What Moves the Score:**
  - GeM pooling; sub-centre / dynamic-margin ArcFace.
  - Query expansion (average query with top neighbours); database-side augmentation.
  - k-reciprocal re-ranking.
  - Flip / multi-scale TTA on embeddings.
  - Local-feature geometric verification of top candidates (landmarks, buildings).
  - Cropping to the object with a detector before embedding.
- **Common Bugs:**
  - Validation split by image instead of by identity → inflated scores.
  - Dot product on unnormalised embeddings.
  - The query itself left in the gallery during validation (trivial rank-1 match).
  - "New identity" threshold tuned on identities seen in training.
  - Memory blow-up from a full `M × N` similarity matrix; compute in chunks.

### Keypoint Detection

**1. Task Description**

Two related problems. **Semantic keypoints:** localise a fixed set of $K$ named points $\{(x_k, y_k, v_k)\}$ per instance (body joints, facial landmarks), with visibility $v_k$. **Interest points:** detect repeatable, distinctive points with descriptors (SuperPoint-style) for matching images and estimating geometry.

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]` (an instance crop for top-down pose).
- Output:
  - Heatmaps `[B, K, H/s, W/s]` (stride s, typically 4) → coordinates `[B, K, 2]` + confidences `[B, K]`.
  - Or direct regression `[B, K, 2]`.
  - Interest points: `[N, 2]` locations + descriptors `[N, D]`.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - Human pose (top-down vs bottom-up), face landmarks, hand pose, animal pose, 3D keypoints, medical landmarks.
  - *Heads:* Gaussian heatmap regression (MSE / KL), direct coordinate regression (L1 / residual log-likelihood), 1D coordinate classification (SimCC).
  - *Interest points + matching:* detector + descriptor heads, then a matcher; the final output is often a homography or relative pose.
- **Olympic AI Framing:**
  - Occluded / invisible keypoints (visibility flags must mask the loss).
  - Crowded multi-person scenes.
  - Unusual domains (animals, sports, X-rays) with small datasets.
  - Scale-normalised metrics: object keypoint similarity (OKS), NME normalised by inter-ocular distance, PCK at a fraction of the bounding-box size.
  - Matching tasks where keypoints are only an intermediate step.

**4. Baseline Pipeline (Pseudo-code)**

```python
def gaussian_heatmaps(kpts, vis, out_hw, sigma=2.0):   # kpts in heatmap coordinates
    h, w = out_hw
    ys, xs = meshgrid(arange(h), arange(w), indexing="ij")
    hm = zeros(K, h, w)
    for k in range(K):
        if vis[k]:
            hm[k] = exp(-((xs - kpts[k, 0]) ** 2 + (ys - kpts[k, 1]) ** 2) / (2 * sigma ** 2))
    return hm

model = load_pretrained_backbone(PATH) + deconv_head(out_channels=K)   # stride-4 heatmaps
for img, kpts, vis in loader(train, transform=affine_aug_with_keypoints(flip_pairs=FLIP_PAIRS)):
    target = gaussian_heatmaps(kpts / STRIDE, vis, (H // STRIDE, W // STRIDE))
    pred = model(img)                                   # [B, K, h, w]
    per_kpt = ((pred - target) ** 2).mean(axis=(2, 3))  # [B, K]
    loss = (per_kpt * vis).sum() / max(vis.sum(), 1)    # invisible keypoints do not contribute
    loss.backward(); opt.step(); opt.zero_grad()

def decode(hm, inv_affine):                             # hm: [K, h, w]
    K, h, w = hm.shape
    idx = argmax(hm.reshape(K, -1), axis=-1)
    y, x = idx // w, idx % w
    xl, xr = clip(x - 1, 0, w - 1), clip(x + 1, 0, w - 1)
    yu, yd = clip(y - 1, 0, h - 1), clip(y + 1, 0, h - 1)
    k = arange(K)
    dx = 0.25 * sign(hm[k, y, xr] - hm[k, y, xl])       # quarter-pixel shift toward the higher neighbour
    dy = 0.25 * sign(hm[k, yd, x] - hm[k, yu, x])
    pts = stack([x + dx, y + dy], axis=-1) * STRIDE
    return apply_affine(inv_affine, pts), hm.reshape(K, -1).max(axis=-1)   # original image coords
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Decide top-down (detect, crop, estimate) vs bottom-up; check visibility annotations and how the metric normalises distance.
  2. Baseline: crops (given or from a detector) + heatmap model at e.g. 256×192.
  3. Add flip TTA, sub-pixel decoding, higher resolution, larger backbones; ensemble by averaging heatmaps.
- **What Moves the Score:**
  - Flip TTA with left/right index swapping (and heatmap alignment).
  - Sub-pixel decoding (quarter-offset, or distribution-aware decoding).
  - Higher heatmap resolution / smaller stride; unbiased coordinate transforms.
  - Strong affine augmentation (rotation ±30°, scale 0.75–1.25).
  - A better person / object detector for top-down pipelines.
- **Common Bugs:**
  - Horizontal flip without swapping symmetric keypoint indices.
  - Crop / resize / padding transforms not inverted at prediction time.
  - Pixel-centre convention (±0.5) creating a systematic offset.
  - Invisible keypoints included in the loss with all-zero targets.
  - `(x, y)` vs `(row, col)` swapped.

### Depth Estimation

**1. Task Description**

Estimate per-pixel depth $D \in \mathbb{R}_{>0}^{H \times W}$ from a single RGB image. Monocular depth is ambiguous in scale, so models predict either **metric** depth (metres) or **relative** depth / inverse depth (correct up to scale and shift).

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]`.
- Output: `[B, 1, H, W]` depth or disparity (inverse depth).
- Ground truth: dense or sparse (LiDAR) depth `[B, 1, H, W]` + validity mask.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - Metric monocular depth (indoor / outdoor); relative / ordinal depth; depth completion (sparse depth + RGB → dense); stereo disparity; joint depth + surface normals.
  - *Losses:* scale-invariant log loss (SILog), L1 on log-depth, scale-and-shift-invariant loss, gradient-matching loss, ordinal regression over depth bins.
- **Olympic AI Framing:**
  - Sparse ground truth → masked losses and metrics.
  - Domain shift (indoor training, outdoor test).
  - Evaluation restricted to a depth range and crop.
  - Unknown camera intrinsics, which break metric scale.
  - Few training images → foundation models plus light fine-tuning.

**4. Baseline Pipeline (Pseudo-code)**

```python
def silog_loss(pred, gt, valid, lam=0.85):
    d = log(pred[valid]) - log(gt[valid])
    return sqrt((d ** 2).mean() - lam * d.mean() ** 2)

def align_scale_shift(pred, gt, valid):                 # solve min_{s,t} ||s * pred + t - gt||^2
    A = stack([pred[valid], ones_like(pred[valid])], axis=-1)
    s, t = least_squares(A, gt[valid])
    return s * pred + t

model = encoder_decoder(load_pretrained_backbone(PATH), out_channels=1, out_activation=softplus)
for img, gt in loader(train, transform=joint_geometric_aug):   # colour aug on img only; hflip ok, no vflip
    valid = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    pred = upsample(model(img), size=gt.shape[-2:], mode="bilinear")
    loss = silog_loss(pred, gt, valid)
    loss.backward(); opt.step(); opt.zero_grad()

def depth_metrics(pred, gt, valid):
    p, g = pred[valid], gt[valid]
    ratio = maximum(p / g, g / p)
    return {"abs_rel": mean(abs(p - g) / g),
            "rmse":    sqrt(mean((p - g) ** 2)),
            "delta1":  mean(ratio < 1.25)}
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Determine whether evaluation is metric or alignment-invariant (median or least-squares scaling per image).
  2. Baseline: zero-shot foundation depth model (e.g. Depth Anything), aligned per image if the evaluation allows.
  3. Fine-tune on competition data with SILog (low encoder LR), then higher resolution, TTA and ensembling.
- **What Moves the Score:**
  - Fine-tuning a depth foundation model rather than training from an ImageNet backbone.
  - Horizontal-flip TTA (un-flip the prediction before averaging).
  - Higher input resolution; sliding windows on large images.
  - Ensembling in log-depth space.
  - Applying exactly the evaluation crop and depth caps during validation.
- **Common Bugs:**
  - Treating a disparity / inverse-depth output as depth.
  - Loss or metric computed on invalid pixels (`gt = 0`); `log(0)`.
  - Resizing sparse ground truth with bilinear interpolation (mixes valid and invalid values); use nearest.
  - Vertical flips or large rotations destroying geometric priors.
  - Unit mismatches: millimetres vs metres; 16-bit PNG depth scaled by 256 or 1000.

### Video Understanding

**1. Task Description**

Map a clip $V \in \mathbb{R}^{T \times 3 \times H \times W}$ to labels, temporal segments, per-frame events or text. This requires both spatial and temporal modelling.

**2. Input & Output Signatures**

- Input: `[B, T, 3, H, W]` (some models expect `[B, 3, T, H, W]`), with T ≈ 8–32 sampled frames.
- Output:
  - Clip classification: `[B, C]`.
  - Per-frame / event spotting: `[B, T, C]` → timestamps.
  - Temporal localisation: `[(t_start, t_end, class, score)]`.
  - Captioning / QA: text.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Action recognition:* clip-level cross-entropy.
  - *Temporal action localisation:* segment proposals + boundary regression.
  - *Event spotting:* per-frame classification + peak detection; mAP within a time tolerance.
  - *Tracking / multi-object tracking:* detection + association.
  - *Video captioning / video QA; anomaly detection.*
- **Olympic AI Framing:**
  - Long untrimmed videos with sparse events.
  - Background frames vastly outnumber event frames.
  - Heavy compute → frame-level features + a lightweight temporal model.
  - Tolerance-based metrics (an event counts within ±δ seconds).
  - Variable frame rate / resolution; corrupted files.

**4. Baseline Pipeline (Pseudo-code)**

```python
def sample_indices(n_frames, T, train):
    if train:                                           # one random frame per equal segment
        edges = linspace(0, n_frames, T + 1).astype(int)
        return [randint(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(T)]
    return linspace(0, n_frames - 1, T).round().astype(int)

# Option A (fast): frozen image backbone, features cached once, light temporal head
feats = {}
for vid in videos:
    frames = decode_frames(vid, sample_indices(num_frames(vid), 32, train=False))   # RGB
    feats[vid] = image_backbone(stack(frames))          # [32, d]
save(feats)

temporal = TransformerEncoder(d, layers=2); cls_head = Linear(d, C)
for f, y in loader(feats):                              # f: [B, 32, d]
    logits = cls_head(temporal(f + positional_embedding).mean(axis=1))   # [B, C]
    loss = cross_entropy(logits, y)
    loss.backward(); opt.step(); opt.zero_grad()

# Option B (accurate, expensive): end-to-end video transformer on [B, T, 3, H, W]
# Inference for both: average logits over several temporal clips and spatial crops per video
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Identify clip classification vs localisation / spotting; check video length, fps and event density.
  2. Baseline: per-frame features from a strong image model + mean pooling + linear head (minutes once features are cached).
  3. Replace mean pooling with a temporal head (transformer / 1D convolution); go end-to-end with a video model only if compute allows.
  4. For spotting: per-frame scores → temporal smoothing → peak picking with temporal NMS.
- **What Moves the Score:**
  - Feature caching, so you can iterate in seconds instead of re-decoding video.
  - Multi-clip, multi-crop TTA.
  - Temporal smoothing + 1D NMS with tolerance-aware thresholds for spotting.
  - Motion cues (frame differences, optical flow) alongside appearance.
  - Concatenating features from several backbones.
- **Common Bugs:**
  - Decoding videos every epoch → GPU idle, data loading is the bottleneck.
  - BGR frames from OpenCV fed to an RGB model.
  - Frame index → timestamp conversion with the wrong (or variable) fps.
  - Clips from the same source video in both train and validation.
  - `[B, T, C, H, W]` vs `[B, C, T, H, W]` layout mismatch.

### Image-to-Text

**1. Task Description**

Conditional language modelling: $p(y \mid I) = \prod_t p(y_t \mid y_{<t}, E(I))$, with a visual encoder $E$ and a text decoder (cross-attention to visual tokens, or visual tokens used as a prefix).

**2. Input & Output Signatures**

- Input: `[B, 3, H, W]` → visual tokens `[B, N, d]`.
- Output: token ids `ℤ^{B×T}` → string (caption, transcription, markup).

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Captioning:* token cross-entropy, optionally followed by CIDEr optimisation via self-critical sequence training.
  - *OCR / text recognition* (printed, scene, handwritten): CTC or attention decoder; character error rate (CER).
  - *Image-to-markup:* formula → LaTeX, chart → table, molecule drawing → SMILES.
  - *Dense captioning; image-to-code.*
- **Olympic AI Framing:**
  - Specialised domains (medical reports, charts, chemistry) where generic captioners fail.
  - Long target sequences.
  - Metric sensitivity to tokenisation and normalisation (casing, spacing, canonical forms).
  - Unusual scripts or vocabularies; handwriting.
  - Rotated, blurred or low-contrast images.

**4. Baseline Pipeline (Pseudo-code)**

```python
vision  = load_pretrained_vision_encoder(V_PATH)        # [B, 3, H, W] -> [B, N, dv]
decoder = load_pretrained_text_decoder(T_PATH)          # cross-attends to visual tokens
proj    = Linear(dv, decoder.dim)

for img, text in loader(train):
    vis = proj(vision(img))                             # [B, N, d]
    ids = tokenize_batch([BOS + t + EOS for t in text], pad_id=PAD_ID)   # [B, T]
    logits = decoder(ids[:, :-1], encoder_states=vis)   # teacher forcing
    loss = cross_entropy(logits, ids[:, 1:], ignore_index=PAD_ID, label_smoothing=0.1)
    loss.backward(); opt.step(); sched.step(); opt.zero_grad()

def describe(img, beams=3, max_len=40):
    vis = proj(vision(img))
    cands = beam_search(decoder, vis, start=BOS, end=EOS, beams=beams,
                        max_len=max_len, length_penalty=1.0, return_all=True)
    valid = [c for c in cands if passes_validity_check(c)]   # e.g. parsable LaTeX / SMILES
    return (valid or cands)[0]
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Identify the target type (free-form captions vs structured strings) and the official scorer.
  2. Baseline: zero-shot pretrained captioner / OCR model / vision-language model.
  3. Fine-tune on competition pairs; raise resolution for text-heavy images.
  4. Tune decoding, then rerank candidates (image-text similarity score, MBR, validity checks).
- **What Moves the Score:**
  - Decoding tuning (beams, length penalty, max length).
  - Candidate reranking / consensus decoding.
  - Domain-appropriate tokenisation (character-level for OCR and chemical strings).
  - Text-preserving augmentation only (no flips; mild rotation / blur / contrast).
  - Validity-checked post-processing with fallback to the next beam.
  - Pseudo-labelling unlabelled images.
- **Common Bugs:**
  - Horizontal flips on images containing text.
  - Padding tokens included in the loss.
  - `max_len` too short, truncating long targets.
  - Evaluating with a different normalisation / tokenisation than the official scorer.
  - Off-by-one teacher-forcing shift between inputs and labels.

## 1.3. Multimodal

### Visual Document Retrieval

**1. Task Description**

Given a text query $q$ and a corpus of document page images $\{P_j\}$, rank pages by relevance. Relevance depends on rendered text, layout, tables, charts and figures, not only on extractable text.

**2. Input & Output Signatures**

- Input: query string; page images `[N, 3, H, W]` (high resolution, often rasterised PDFs).
- Page representations:
  - OCR text per page.
  - Single vector `[N, d]`.
  - Multi-vector (late interaction) `[N, n_patches, d]`.
- Output: ranked page ids `[Q, k]`.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *OCR pipeline:* OCR → sparse / dense text retrieval.
  - *Single-vector multimodal embeddings:* image and text in one space; contrastive (InfoNCE) training.
  - *Late interaction* (ColPali-style): per-token query vectors vs per-patch page vectors, MaxSim scoring, contrastive loss.
  - *Reranking* top candidates with a vision-language model (pointwise relevance).
  - *Cross-lingual retrieval; chart- / table-centric queries; document- vs page-level granularity.*
- **Olympic AI Framing:**
  - Visually rich pages (charts, infographics, tables) where OCR loses the answer.
  - Large corpora under GPU-memory limits (multi-vector indexes are big).
  - Queries in a different language from the documents.
  - Relevance labelled per page while documents span many pages.
  - nDCG@5-style metrics that reward getting the top few right.

**4. Baseline Pipeline (Pseudo-code)**

```python
# Path A: OCR -> text retrieval (robust, cheap)
page_text = [ocr(p, lang=DOC_LANG) for p in pages]
text_ranks = {q: hybrid_retrieve(q, page_text) for q in queries}     # BM25 + dense + RRF (see Text Ranking)

# Path B: late interaction over page patches
page_emb = [l2_normalize(vlm_page_encoder(p)) for p in pages]        # each [n_patches, d]

def maxsim(q_tok, page):                                # q_tok [n_q, d], page [n_p, d]
    return (q_tok @ page.T).max(axis=1).sum()           # best patch per query token, summed

def search(query, k=5, k_pre=100):
    q_tok = l2_normalize(vlm_query_encoder(query))      # [n_q, d]
    pooled = stack([e.mean(axis=0) for e in page_emb])  # cheap first pass: [N, d]
    cands = topk_indices(pooled @ q_tok.mean(axis=0), k_pre)
    visual_rank = sorted(cands, key=lambda j: -maxsim(q_tok, page_emb[j]))
    return reciprocal_rank_fusion([visual_rank, text_ranks[query]])[:k]
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Inspect what pages look like (text-heavy vs visual) and what queries ask.
  2. Baseline: OCR + BM25, which sets a robust floor.
  3. Add single-vector multimodal embeddings, then late-interaction models if provided.
  4. Fuse rankings; rerank the top-k with a vision-language model if time allows.
- **What Moves the Score:**
  - RRF fusion of OCR-text and visual rankings.
  - Higher rasterisation resolution for small fonts.
  - Token pooling / fp16 or quantised storage for multi-vector indexes.
  - Contrastive fine-tuning with hard negatives drawn from the same document.
  - Query translation or rewriting for cross-lingual queries.
- **Common Bugs:**
  - Low-resolution rasterisation destroying small text.
  - OCR language pack mismatch (e.g. losing Vietnamese diacritics).
  - Submitting document ids when page ids are expected (or the reverse).
  - fp32 multi-vector storage exhausting memory.
  - Query and page encoders swapped (asymmetric encoders / prefixes).

### Document Question Answering

**1. Task Description**

Given a document image (or several pages) $P$ and a question $q$, output an answer string $a$, either extracted from the document's text or generated. Combines text, 2D layout and visual cues.

**2. Input & Output Signatures**

- Input: page image `[3, H, W]` + question tokens; for layout-aware models also OCR words with boxes `[(w_i, [x0, y0, x1, y1])]`, boxes normalised to 0–1000.
- Output: answer string. Extractive models: start / end logits over OCR sub-tokens, `ℝ^{T}` each.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Extractive, layout-aware* (LayoutLM-style): span heads over OCR tokens.
  - *Generative, OCR-free* (Donut / Pix2Struct-style): image → answer tokens.
  - *Key information extraction:* fixed fields (invoice total, date) as token classification over OCR tokens.
  - *Table / chart QA:* numeric reasoning and aggregation.
  - *Multi-page QA:* page retrieval + single-page QA.
- **Olympic AI Framing:**
  - Poor scans, handwriting, skewed or rotated pages.
  - Answers that require arithmetic and are not present verbatim.
  - Documents longer than the model's context.
  - ANLS scoring with a 0.5 cut-off.
  - Non-English documents and no OCR engine available offline.

**4. Baseline Pipeline (Pseudo-code)**

```python
def anls(pred, golds, tau=0.5):                         # average normalised Levenshtein similarity
    best = 0.0
    for g in golds:
        p, g = normalize(pred), normalize(g)            # lowercase, collapse whitespace
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best

# Extractive, layout-aware
words, px_boxes = ocr_with_boxes(page)
boxes = [[int(1000 * x0 / W), int(1000 * y0 / H), int(1000 * x1 / W), int(1000 * y1 / H)]
         for x0, y0, x1, y1 in px_boxes]                # normalise to 0-1000
ids, bbox, word_ids = tokenize_with_layout(question, words, boxes, max_len=512)   # one box per sub-token
start_logits, end_logits = layout_model(ids, bbox, pixel_values=resize(page))
answer = best_valid_span(start_logits, end_logits, word_ids, words, max_answer_words=20)

# Generative, OCR-free
answer = ocr_free_model.generate(image=page, prompt=f"<question>{question}</question><answer>")
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Measure answer coverage: how often does the gold answer appear verbatim in the OCR text? High → extractive; low → generative.
  2. Check OCR quality on a sample; run a pretrained document-QA model zero-shot and compute ANLS on validation.
  3. Fine-tune; ensemble extractive and generative models, selecting by confidence; normalise answer formats.
- **What Moves the Score:**
  - Better OCR (or an OCR ensemble). This is often the biggest lever for layout-aware models.
  - Higher input resolution for OCR-free models.
  - Normalising answers to the training format (dates, currency, thousands separators).
  - Page retrieval before QA for multi-page documents.
  - Confidence-based selection among models, with fallbacks.
- **Common Bugs:**
  - Boxes not normalised to 0–1000, or x / y swapped.
  - Wrong OCR reading order on multi-column layouts.
  - OCR engine missing on the offline machine → the pipeline crashes at inference.
  - ANLS normalisation that differs from the official scorer.
  - Truncation beyond 512 tokens dropping the answer region.

### Visual Question Answering

**1. Task Description**

Learn $p(a \mid I, q)$: answer a natural-language question about an image, with the answer chosen from a fixed vocabulary (classification) or generated as free text.

**2. Input & Output Signatures**

- Input: image `[B, 3, H, W]`, question ids `ℤ^{B×T}`.
- Output: logits over an answer vocabulary `[B, A]` (A ≈ a few thousand), or generated answer tokens.

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Open-ended short answer:* classification over the top-A answers with soft-score BCE targets.
  - *Multiple choice:* score each (image, question, option) triple.
  - *Yes/No; counting* (classification or regression).
  - *Text-reading VQA:* OCR tokens as extra input.
  - *Knowledge-based VQA; medical / remote-sensing / chart VQA.*
- **Olympic AI Framing:**
  - Strong language priors (e.g. "yes" dominates yes/no questions); test sets may be rebalanced against them.
  - Counting and spatial-reasoning questions.
  - Specialised imagery with small datasets.
  - Several annotator answers per question → soft targets and the VQA accuracy metric $\min(\#\text{matches}/3, 1)$.

**4. Baseline Pipeline (Pseudo-code)**

```python
def soft_target(answers, vocab):                        # VQA-accuracy-style soft labels
    t = zeros(len(vocab))
    for a, n in count(normalize_answer(a) for a in answers).items():
        if a in vocab:
            t[vocab[a]] = min(n / 3, 1.0)
    return t

img_enc = load_pretrained_vision_encoder(V_PATH)        # -> [B, N, d]
txt_enc = load_pretrained_text_encoder(T_PATH)          # -> [B, T, d]
fusion  = CrossAttentionBlocks(d, layers=2)             # text attends to image
head    = MLP(d, len(vocab))

for img, q, answer_lists in loader(train):
    h = fusion(query=txt_enc(tokenize(q)), context=img_enc(img))[:, 0]   # [B, d]
    targets = stack([soft_target(a, vocab) for a in answer_lists])       # [B, A]
    loss = binary_cross_entropy_with_logits(head(h), targets)
    loss.backward(); opt.step(); opt.zero_grad()

# Generative alternative with a vision-language model
answer = normalize_answer(vlm.generate(image=img,
                                       prompt=f"Question: {q}\nAnswer with one word or number:"))
```

**5. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Analyse answers: vocabulary size and coverage, distribution by question type (yes/no, number, other).
  2. Train question-only and image-only baselines to measure bias; run a pretrained VQA model zero-shot.
  3. Fine-tune a classifier or vision-language model; break the validation score down by question type.
  4. Route weak question types to specialists (counting, OCR); ensemble.
- **What Moves the Score:**
  - Soft-target BCE over the answer vocabulary.
  - Answer normalisation identical to the metric's.
  - Question-type routing to specialist models or prompts.
  - Higher resolution for fine details; OCR tokens for text-reading questions.
  - Logit ensembling; constrained decoding to the answer vocabulary for generative models.
- **Common Bugs:**
  - Normalisation mismatch (`"two"` vs `"2"`, articles, punctuation).
  - Answer vocabulary too small, so many gold answers are unreachable (measure coverage).
  - The same image appearing in train and validation with different questions (split by image).
  - Augmentations that change the answer (horizontal flip for left/right questions; crops removing counted objects).
  - Verbose generative outputs (`"The answer is 2."`) scored as wrong.

# 2. General Tips

## 2.1. Data

## 2.2. Trick & Technique

## 2.3. LLM Usage

## 2.5. Strategy 

### What to Do First

1. Read the rules.
2. Organize the final submission folder and ensure the reproducibility of the results. If model weights are required, make sure to save your model weights after training and load them when running inference on the test set.
3. Determine the list of provided pretrained models.

### Device Distribution & Time Management

Each team will be provided with 2 computers and 6 hours to train the best model. Therefore, a strategy to maximize the usage of both computers and time is necessary. Ideally, we suggest following this workflow:

**First 5 hours:**

- Dedicate 1 computer to data analysis and processing, while using the other computer to analyze the baseline model and results.
- After completing data processing, train the baseline model to see if the results improve, which indicates that the data has been effectively processed.
- Next, improve the architecture and training paradigm to achieve the best score.

**The last 1 hour:** Focus only on fine-tuning hyperparameters, which is the safest approach (in case you do not know how to code just like me). Additional data processing is also a good option, though only advanced techniques should be used here since everything essential should already be done within the first 5 hours.