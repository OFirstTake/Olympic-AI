# Olympic AI Guide

## Table of Contents

<a href="#1-foundational-task">1. Foundational Task</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-natural-language-processing">1.1. Natural Language Processing</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#feature-extraction">Feature Extraction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#fill-mask">Fill-Mask</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#text-classification">Text Classification</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#token-classification">Token Classification</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#sentence-similarity">Sentence Similarity</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#text-ranking">Text Ranking</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#question-answering">Question Answering</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#text-generation">Text Generation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#12-computer-vision">1.2. Computer Vision</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#image-feature-extraction">Image Feature Extraction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#image-classification">Image Classification</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#object-detection">Object Detection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#image-segmentation">Image Segmentation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#image-retrieval">Image Retrieval</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#keypoint-detection">Keypoint Detection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#depth-estimation">Depth Estimation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#video-understanding">Video Understanding</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#image-to-text">Image-to-Text</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#13-multimodal">1.3. Multimodal</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#visual-document-retrieval">Visual Document Retrieval</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#document-question-answering">Document Question Answering</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#visual-question-answering">Visual Question Answering</a><br>
<a href="#2-general-tips">2. General Tips</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-data">2.1. Data</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-trick--technique">2.2. Trick &amp; Technique</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-llm-usage">2.3. LLM Usage</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#24-version-control">2.4. Version Control</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#25-strategy">2.5. Strategy</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#what-to-do-first">What to Do First</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226;&nbsp; <a href="#device-distribution--time-management">Device Distribution &amp; Time Management</a>

# 1. Foundational Task

## 1.1. Natural Language Processing

### Feature Extraction

**1. Task Description**

Turn each text into a fixed-length list of numbers (an *embedding*) using a pretrained language model, usually kept frozen. Texts with similar meaning get similar embeddings. The embedding is not the final answer: it is the input to a simple, fast model (logistic regression, gradient-boosted trees, nearest-neighbour search, clustering) that makes the actual prediction.

Core idea: let a large pretrained model do the "understanding" once, then spend your time on the cheap model on top.

**2. Input & Output**

- Input: raw text (sentences, paragraphs or whole documents).
- Output: one embedding per text (typically a few hundred to ~1,000 numbers), usually made by averaging the model's per-token outputs and scaling the vector to length 1.
- What gets submitted: the predictions of the downstream model trained on these embeddings.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Some words in a sentence are hidden behind a special mask token, and the model must guess them from the surrounding context. BERT-style models are pretrained exactly this way, so they can do it out of the box. In competitions it appears in three forms:
- Predicting missing words directly.
- **Domain-adaptive pre-training (DAPT):** continuing this "guess the hidden word" training on unlabelled in-domain text, so the model learns the domain before being fine-tuned on the real task.
- **Cloze-style classification:** rewriting a classification task as fill-in-the-blank (e.g. *"The movie was [MASK]."* → "great" or "awful") so it works with zero or few labels.

**2. Input & Output**

- Input: text with one or more masked positions, or a sentence with a blank plus a list of candidate answers.
- Output: for each masked position, a ranked list of likely words with probabilities; with candidates, the best-scoring candidate.
- Cloze classification: one score per class, taken from the probability of that class's label word.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Assign a label to a piece of text, or to a pair of texts. The label can be one class out of several (topic, sentiment), several labels at once (tags), or an ordered score (a 1–5 rating, an essay grade). The standard approach is to fine-tune a pretrained encoder with a small classification layer on top.

**2. Input & Output**

- Input: a text, or two texts joined with a separator (e.g. premise + hypothesis).
- Output: a probability per class (exactly one class), an independent yes/no probability per label (several labels), or a single number (rating / regression).

**3. Task Variants & Olympic AI Framing**

- **Common Sub-tasks:**
  - *Binary:* 1 logit + binary cross-entropy (BCE); tune the decision threshold.
  - *Multi-class:* one score per class + cross-entropy (optionally label smoothing).
  - *Multi-label:* an independent score per label + per-label BCE (or focal / asymmetric loss); per-class thresholds.
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

**4. Strategy, Score Boosters & Pitfalls**

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

Give a label to every word in a text. It is mostly used to find **spans**: names, places, personal information, parts of an essay. Spans are written as per-word tags: `B` = beginning of a span, `I` = inside a span, `O` = outside any span (e.g. *"Barack/B-PER Obama/I-PER visited/O Hanoi/B-LOC"*). The model predicts a tag for each word, and the tags are converted back into spans.

**2. Input & Output**

- Input: text, either pre-split into words or as raw characters. Because the model works on sub-word pieces, you need a mapping from each piece back to its word or character position.
- Output: one tag per word, converted into a list of spans (start position, end position, type).

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

**4. Strategy, Score Boosters & Pitfalls**

- **Problem-Solving Strategy:**
  1. Establish the annotation unit (characters vs words) and the metric (entity- vs token-level).
  2. Baseline: encoder + linear token head, first-sub-token labelling, sliding windows over long texts.
  3. Add constrained decoding (CRF or Viterbi with BIO transition constraints), larger encoders, then ensembling by averaging *word-level* probabilities.
  4. Trade precision against recall with a threshold on the `O` probability.
- **What Moves the Score:**
  - `O`-threshold: predict an entity whenever the model's probability for `O` falls below a threshold tuned on OOF. This is crucial for recall-weighted metrics.
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

Score how close two texts are in meaning. There are two ways to build the scorer:
- **Bi-encoder:** embed each text separately and compare the two vectors with cosine similarity. Fast, and texts can be pre-indexed, so it scales to search over millions of texts.
- **Cross-encoder:** feed both texts into the model together and let it output a score directly. More accurate, but it has to run once for every pair.

**2. Input & Output**

- Input: a pair of texts, sometimes with an extra context field (e.g. a topic or category).
- Output: a similarity score (e.g. 0–5), a probability that the texts are paraphrases, or a 3-way label (entailment / neutral / contradiction).

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

**4. Strategy, Score Boosters & Pitfalls**

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

Given a query and a large collection of documents, return the most relevant documents in the right order. The standard design has two stages:
1. **Retrieve:** a fast method (keyword search such as BM25, embedding search, or both) picks a few hundred candidates from the whole collection.
2. **Rerank:** a slower, more accurate model (a cross-encoder that reads the query and document together) reorders only those candidates.

**2. Input & Output**

- Input: query text and a document collection (long documents are usually split into passages).
- Output: for each query, an ordered list of the top-k document ids, optionally with scores.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Given a question and a context passage, find the answer. In **extractive QA** (the most common form) the answer is a piece of text copied from the passage: the model predicts where the answer starts and where it ends. If the passage does not contain the answer, the model should return "no answer". Other forms: multiple choice, open-domain QA (first find the right passage, then read it), and generative QA (write the answer freely).

**2. Input & Output**

- Input: question + context. Long contexts are split into overlapping windows so nothing is cut off.
- Output: the answer text (a substring of the context, or empty for "no answer"); for multiple choice, the chosen option.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Produce output text from input text: summaries, translations, answers to instructions, or structured output such as JSON. The model writes one token at a time, each based on the input and on what it has written so far. Two choices dominate the result:
- **How the model is adapted:** plain prompting, few-shot examples in the prompt, or light fine-tuning (e.g. LoRA).
- **How text is decoded:** greedy, beam search or sampling, plus length limits and stop rules.

**2. Input & Output**

- Input: source text or a prompt.
- Output: generated text, sometimes parsed into a required format (JSON, a label, code).

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

**4. Strategy, Score Boosters & Pitfalls**

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

Turn each image into a fixed-length vector (one per image) or a grid of vectors (one per image region) using a pretrained vision model. As with text embeddings, the vectors are not the final answer: they feed a simple model (linear classifier, nearest-neighbour search, clustering, tree model).

**2. Input & Output**

- Input: images, resized and normalised the way the pretrained model expects.
- Output: one vector per image (a global description), or one vector per patch / region (dense features) for tasks that need spatial detail.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Assign a label to a whole image: one class, several labels at once, or a grade / score (e.g. disease severity). The standard approach is to fine-tune a pretrained CNN or vision transformer with a new classification layer.

**2. Input & Output**

- Input: an image.
- Output: a probability per class (exactly one class), an independent probability per label (several labels), or a single number (grade / regression).

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

**4. Strategy, Score Boosters & Pitfalls**

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

Find every object of interest in an image. For each object, output a rectangle around it (a *bounding box*), its class, and a confidence score. The number of objects varies from image to image and can be zero.

**2. Input & Output**

- Input: an image. Training labels are a list of boxes with a class for each.
- Output: per image, a list of (box corners, class, confidence), in the original image's pixel coordinates.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Label an image pixel by pixel.
- **Semantic segmentation:** every pixel gets a class (road, sky, car); all cars share the same label.
- **Instance segmentation:** each object gets its own mask (car #1, car #2).
- **Panoptic segmentation:** both at once: background regions by class, countable objects as separate instances.

**2. Input & Output**

- Input: an image.
- Output: a mask the same size as the image with a class for every pixel (semantic), a foreground probability per pixel (binary), or a list of (mask, class, confidence) (instance).
- Submissions are often masks compressed as run-length encoding (RLE) strings.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Given a query image, find the images in a gallery that show the *same thing*: the same product, landmark, person or animal. A model is trained so that images of the same identity get nearby embeddings and different identities get distant ones (*metric learning*). Retrieval is then a nearest-neighbour search over those embeddings.

**2. Input & Output**

- Input: query images and a gallery of images.
- Output: for each query, the top-k gallery ids ranked by similarity. In open-set problems, a special "new / unknown" answer is returned when no gallery image is close enough.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Two related problems:
- **Semantic keypoints:** locate a fixed set of named points on each object (body joints, facial landmarks), each with a visible / hidden flag. The usual approach is to predict one heatmap per keypoint and take its peak as the location.
- **Interest points:** find distinctive points that can be matched between two images of the same scene, used for image stitching or estimating camera motion.

**2. Input & Output**

- Input: an image, or a crop around one person / object.
- Output: an (x, y) location and a confidence for each keypoint. For interest points: a list of point locations, each with a descriptor vector used for matching.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Predict how far each pixel is from the camera, from a single image. One image cannot reveal absolute scale (a small nearby object looks like a large distant one), so models predict either **metric depth** (real distances in metres) or **relative depth** (correct ordering and shape, but unknown scale and offset).

**2. Input & Output**

- Input: an RGB image.
- Output: a depth map the same size as the image (sometimes inverse depth).
- Ground truth may be dense, or sparse (only some pixels have a value, e.g. from LiDAR), with a mask marking which pixels are valid.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Understand what happens in a video: classify a clip, find when actions start and end, spot the exact moment an event happens, or describe it in text. It needs both appearance (what is in each frame) and time (how things change between frames). A common, cheap approach: extract features from sampled frames with an image model, then put a light temporal model on top.

**2. Input & Output**

- Input: a video, usually sampled down to a fixed number of frames (around 8–32).
- Output: a class for the clip; per-frame scores turned into event timestamps; time segments (start, end, class, score); or text.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Generate text from an image: captions, transcriptions of written text (OCR), or structured strings such as LaTeX from a formula image or SMILES from a molecule drawing. A vision encoder reads the image, and a text decoder writes the output one token at a time while looking at the image.

**2. Input & Output**

- Input: an image.
- Output: a text string (caption, transcription or markup).

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

**4. Strategy, Score Boosters & Pitfalls**

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

Given a text query and many document pages stored as images (scanned or rendered PDFs), find the most relevant pages. Relevance can depend on text, layout, tables, charts and figures, so plain text extraction may miss information. Two main approaches: run OCR and then use text retrieval, or use a vision-language model that embeds the page image directly.

**2. Input & Output**

- Input: a query text and page images (high resolution).
- Page representation: OCR text, one vector per page, or many vectors per page (one per image patch).
- Output: for each query, a ranked list of page ids.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Answer a question about a document image (invoice, form, report, multi-page PDF). The answer is either copied from text on the page (extractive) or generated (when it needs reformatting or a calculation). Good models combine three signals: what the words say, where they sit on the page, and how the page looks.

**2. Input & Output**

- Input: page image(s) + a question. Layout-aware models also take the OCR words together with their positions on the page.
- Output: the answer as a string.

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

**4. Strategy, Score Boosters & Pitfalls**

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

Answer a natural-language question about an image (*"How many dogs are there?"*, *"What colour is the car?"*). Either pick the answer from a fixed list of common answers (classification), or generate it as text with a vision-language model.

**2. Input & Output**

- Input: an image + a question.
- Output: one short answer: a word, a number, yes/no, or a short phrase.

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
  - Several annotator answers per question → soft targets, and the VQA accuracy metric (an answer counts as fully correct if at least 3 annotators gave it).

**4. Strategy, Score Boosters & Pitfalls**

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

From my observation, data processing is often the deciding factor in winning these types of competitions. I highly recommend creating a standardized data workflow and practicing it beforehand so your team can execute it quickly and effectively. While workflows may vary, there are two must-have steps:

**Exploratory Data Analysis (EDA):**

This step helps you gain deep insights into the provided dataset before writing any model code.

- **Sample Content:** Identify unique patterns within raw samples from each class. These insights can guide your choice of techniques or model architecture. Be highly alert for data leakage (e.g., background elements that allow the model to cheat, or special text tokens that inadvertently leak the target label).

- **Sample Shape:** Check if the data dimensions are uniform. For images, check the width and height distributions. For text, analyze the sequence lengths. For video, in addition to spatial dimensions, check the total frame count and FPS.

- **Data Distribution & Imbalance:** Analyze the label distribution to identify severe class imbalance or long-tailed data classes.

- **Missing Values & Corrupted Data:** Locate all null/NaN values, broken images, and corrupted files so they don't crash your training loop.

- **Outliers & Anomalies:** Handle abnormal samples properly (e.g., bounding boxes lying outside image boundaries, or text sequences that vastly exceed standard token limits).

**Data Preprocessing & Cleaning:**

Usually, a basic data preprocessing pipeline is provided with the baseline code. Therefore, you only need to adapt it to your specific strategy (just remember not to overcomplicate it without understanding the data). A complete pipeline typically includes:

- **Computer Vision (CV):**

    - **Resizing & Cropping:** Rescale or tile the images to match the required input resolution of your model.

    - **Normalization:** Apply the specific mean and standard deviation of the dataset used to pretrain the model (e.g., ImageNet stats) to ensure optimal convergence.

    - **Format Conversion:** Convert between color spaces (e.g., BGR to RGB, or Grayscale) as required by the architecture.

    - **Data Augmentation:** Prevent overfitting and improve model generalization.

        - **Standard:** Flips, Rotations, Color Jitter.

        - **Advanced:** CutMix, MixUp, Mosaic.

- **Natural Language Processing (NLP):**

    - **Text Cleaning:** Strip HTML tags, remove extra whitespace or emojis, and fix basic typos.

    - **Sequence Truncation & Padding:** Truncate or pad the text intelligently (e.g., keeping only the head, only the tail, or head+tail) to retain crucial information while fitting within context limits.

    - **Data Augmentation:** Apply synonym replacement, random token masking, back-translation, or add noise directly at the embedding level (e.g., AEDA).

## 2.2. Trick & Technique

There are several tricks and techniques that usually give you a huge advantage in these kinds of competitions:

**Metric Optimization:** Always optimize your model and design your solution specifically for the given evaluation metric.

**Ensembling:** Combining several models almost always beats the single best model, as long as they make different mistakes. This means we want them to learn different patterns from the dataset, which can be achieved by making them slightly different from each other, such as changing hyperparameters, using different pretrained backbones or training each model on a different subset of the data. Once we have them, we can combine their outputs using:

  - **Averaging:** Average the predicted probabilities (classification) or outputs (regression). Use a weighted average when one model is clearly stronger.
  - **Voting:** Majority vote over the predicted labels. Useful when models only output hard labels.
  - **Rank Averaging:** Average the ranks instead of the raw scores when models are poorly calibrated or the metric is rank-based (AUC, retrieval, ranking).
  - **Task-Specific Fusion:** For detection, merge boxes with Weighted Boxes Fusion (WBF) instead of NMS. For retrieval, fuse ranked lists with Reciprocal Rank Fusion (RRF).

In short-duration competitions, we generally prefer ensembling smaller, diverse models over training a single super-heavy model, as it enables faster iteration while delivering a higher score.

**Free Gains (Single Model):**

  - **Test-Time Augmentation (TTA):** Predict on a few augmented copies of each test sample (e.g., horizontal flip, multi-scale, several crops) and average the outputs. Only use augmentations that do not change the label (no flips for left/right questions or text in images).
  - **Weight Averaging:** Keep an EMA of the weights during training, or average the last few checkpoints (SWA). It usually beats the single best checkpoint and costs nothing at inference.
  - **Threshold Tuning:** For F1, multi-label, or imbalanced tasks, tune the decision threshold (per class if needed) on the validation set instead of using 0.5.

## 2.3. LLM Usage

Each team will be provided with access to a Large Language Model (LLM) to help brainstorm ideas, implement code, and debug. However, there are a few constraints that should be considered:

- There will be a limit on the context window (usually around 2,000 tokens).
- The LLM is only available for the first 5 hours and will be turned off for the last 1 hour.

The first constraint shapes how we should use the LLM, while the second defines our time management strategy which will be discussed in the section below. We will focus only on the first constraint here.

In terms of brainstorming, if a task is unfamiliar, the LLM is a great way to kick things off. You can use it to identify the type of task, ask for recommended approaches and figure out what needs to be done to solve the problem. 

For coding, the context window limitation is a strict constraint. Because of it, you cannot ask the LLM to generate large code all at once. Therefore, it is much better to do things like:

1. Generate an outline for the end-to-end pipeline.
2. Generate code for each function/component individually (1 session for each).
3. Debug specific code snippets if you run into errors.

## 2.4. Version Control

Version control is recommended to track progress, document changes and experiments, and ensure result reproducibility. There are a few great methods to start with:

- **Naming Convention:** One recommended method is saving your notebook after each change and run using a naming format such as `<score>_<applied_techniques/hyperparameters>` (e.g., `0.842_DataAugmentation.ipynb`).
- **Sync Weights with Notebooks:** Always save the model checkpoints with the exact same name as the notebook to easily match the best weights for final inference (e.g., `0.842_DataAugmentation.pth` for `0.842_DataAugmentation.ipynb`).
- **Keep a Quick Log:** Maintain a brief note to track what you have tried and avoid duplicating experiments.

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
- Next, improve the architecture and training paradigm to achieve the best score (remember to use version control for better tracking and reproducibility).

**The last 1 hour:** Focus only on fine-tuning hyperparameters, which is the safest approach (in case you do not know how to code just like me). Additional data processing is also a good option, though only advanced techniques should be used here since everything essential should already be done within the first 5 hours.
