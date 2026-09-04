
# NLP — The Complete Guide

### Everything you need to walk into an interview confident

---

## HOW TO USE THIS GUIDE

This guide moves from **simple → complex**, exactly the way an interviewer will probe you. Start with what NLP is, then how words become numbers, then the models that reason about those numbers, then the architectures, then the cutting-edge stuff.

**Every concept has:**
- A plain-English explanation
- A concrete example
- Where it's used in real products

---

---

# PART 1: NLP FUNDAMENTALS

---

## What is NLP?

NLP = Natural Language Processing. Getting computers to understand, interpret, and generate human language.

**The core problem:** Computers only understand numbers. Language is messy, ambiguous, and infinite. NLP is the bridge.


**Why language is hard for machines:**

"Bank" = river bank OR financial institution — meaning depends on context.
"I saw a man with a telescope" — who has the telescope? Ambiguous.
"Oh great, another Monday" — sarcasm. The opposite of the literal meaning.

---

## The NLP Pipeline — Every System Does These Steps

```
Raw Text → Tokenisation → Encoding → Model → Output
```

**Step 1: Tokenisation** — Split text into units the model can process.

```
"I love NLP"  →  ["I", "love", "NLP"]

Subword tokenisation (used by BERT/GPT):
"unhappiness"  →  ["un", "##happi", "##ness"]
```

Why subword? So the model handles rare or unseen words without breaking.

**Step 2: Stopword Removal** — Remove words that add no meaning ("the", "a", "is"). Used in search engines and older NLP systems. Not used in BERT/GPT — they need all words for context.

**Step 3: Stemming vs Lemmatisation**

Stemming = crude cut: "running" → "runn" (might not be a real word)
Lemmatisation = smart root: "running" → "run" (always a real word)

**Step 4: Encoding** — Convert tokens into numbers (covered next section).

**Step 5: Model** — BERT, GPT, LSTM, etc. — covered in Part 3+.

---

## Core NLP Tasks — Know These for Interviews

| Task | What It Does | Product Example |
|---|---|---|
| Text Classification | Is this spam/positive/relevant? | LinkedIn spam filter |
| Named Entity Recognition (NER) | Extract "Apple" = company, "London" = city | Resume parsing |
| Sentiment Analysis | Positive / Negative / Neutral | Product review analysis |
| Machine Translation | English → French | LinkedIn multilingual feed |
| Question Answering | "Who founded LinkedIn?" → "Reid Hoffman" | Search engines |
| Text Generation | Complete or create new text | ChatGPT, LinkedIn AI writer |
| Summarisation | Shorten a long document | Meeting notes, article TL;DR |
| Semantic Search | Find meaning, not just keyword match | LinkedIn job search |

---


---

# PART 2: WORD REPRESENTATIONS — FROM SIMPLE TO SMART

> The most important question in NLP: how do you turn words into numbers that preserve meaning?


---

## 1. One-Hot Encoding

**Idea:** Give each word a unique slot in a giant list. Put 1 in that slot, 0 everywhere else.

**Example — Vocabulary: [cat, dog, fish, bird]**

```
"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"fish" → [0, 0, 1, 0]
"bird" → [0, 0, 0, 1]
```

**The problems:**

With 100,000 words → each word is a list of 100,000 numbers (99,999 are zero). Enormous and wasteful.

"cat" and "dog" are both animals, but their vectors are exactly as different from each other as "cat" and "fish." The model has zero knowledge of similarity.

**When it's still used:**

Small, fixed category lists (yes/no, category labels). Simple classification where vocabulary is tiny.

---

## 2. Bag of Words (BoW)

**Idea:** Count how many times each word appears in a document. Ignore word order.

**Example:**
```
Sentence: "I love dogs, dogs are great"
Vocabulary: [I, love, dogs, are, great]
Vector:     [1,  1,    2,    1,   1  ]
```

**Problem:** "The dog bit the man" and "The man bit the dog" produce the same vector. Order is gone.

**When used:** Document classification, spam filters, TF-IDF search (covered below).

---

## 3. TF-IDF (Term Frequency – Inverse Document Frequency)

**Idea:** A word is important if it appears often in THIS document but rarely across ALL documents.

```
TF  = how often word appears in this document
IDF = log(total documents / documents containing this word)
TF-IDF = TF × IDF
```

**Example:**
"Machine" appears in 90% of AI articles → low IDF (not distinctive).
"Backpropagation" appears in 5% → high IDF (distinctive, meaningful).

**Where used:** Classic search engines, document similarity, keyword extraction.

---

## 4. Word2Vec (2013, Google)

**The breakthrough idea:** "You shall know a word by the company it keeps." Words that appear in similar contexts have similar meanings. Train a small neural net to predict nearby words — the weights it learns become the word vectors.

**Two training methods:**

CBOW (Continuous Bag of Words) = predict the middle word from surrounding words.
```
["The", ___, "sat", "on"] → predict "cat"
```

Skip-gram = predict surrounding words from the middle word.
```
"cat" → predict ["The", "sat", "on"]
```

**Result:** Every word gets a dense vector of ~300 numbers.

```
"king"  → [0.82, 0.14, 0.65, 0.33, ... 300 numbers]
"queen" → [0.79, 0.18, 0.61, 0.38, ... 300 numbers]
"dog"   → [0.21, 0.76, 0.12, 0.88, ... 300 numbers]
```

Similar words end up with similar numbers. That's the magic.

**The famous arithmetic:**
```
king − man + woman ≈ queen
paris − france + italy ≈ rome
```

The model learned gender and royalty and geography as geometric directions in vector space — without anyone telling it what those concepts were.

**The fatal flaw:** Every word gets ONE fixed vector regardless of context.

```
"river bank" → bank gets the SAME vector as in "national bank"
```

Word2Vec cannot disambiguate. Context doesn't change the vector.

**Where used:** LinkedIn's older search and matching systems, document clustering, word similarity.

---

## 5. GloVe (2014, Stanford)

Global Vectors for Word Representation. Similar to Word2Vec but trains on co-occurrence statistics across the entire corpus rather than local windows.

"ice" appears near "cold" and "water" often → their vectors are close.
"steam" also appears near "hot" and "water" → steam is close to water but opposite to ice.

GloVe captures global patterns that Word2Vec might miss.

**Difference from Word2Vec:** Word2Vec trains on local windows. GloVe trains on global co-occurrence matrix. Both produce static embeddings.

---

## 6. FastText (2016, Meta)

Word2Vec treats "run" and "running" as completely separate words. FastText breaks words into character n-grams:

```
"running" → ["run", "runn", "runni", "runnin", "running", "unning", ...]
```

The word's embedding is the average of all its n-gram embeddings.

**Why this matters:** It can handle typos and unseen words. "runningg" would still get a reasonable embedding because most of its subwords exist in the vocabulary.

**Where used:** Languages with rich morphology (German, Finnish), misspelling-tolerant search.

---

## 7. Contextual Embeddings — BERT Era (2018+)

**The fix for Word2Vec's flaw:** The SAME word gets a DIFFERENT vector depending on the sentence.

```
"I went to the river bank"
 bank → [0.2, 0.9, 0.1 ...]   ← nature/geography direction

"I went to the bank for a loan"
 bank → [0.8, 0.1, 0.9 ...]   ← finance/institution direction
```

The model reads the WHOLE sentence before deciding what each word means. This is what BERT and GPT do — they produce contextual embeddings.

**Summary comparison:**

```
One-Hot    → identity, no meaning, huge
Word2Vec   → meaning, static (one vector per word)
GloVe      → global meaning, static
FastText   → handles subwords, static
BERT/GPT   → contextual meaning, different vector per context ← state of the art
```

---


---

# PART 3: SEQUENCE MODELS — RNN, LSTM, GRU

> Before Transformers, these were the best tools for understanding sequences (text, time-series).

**Why sequence models exist:** Word order matters enormously.
"The dog bit the man" ≠ "The man bit the dog"

A model that reads words one by one, remembering what came before, can capture this. That's exactly what RNNs do.


---

## 1. RNN — Recurrent Neural Network

**Idea:** Process one word at a time. After each word, pass a "hidden state" h (the model's memory) to the next step.

```
"The" → [RNN cell] → h₁
              ↓
"dog" → [RNN cell] → h₂   (uses h₁ + "dog")
              ↓
"ate" → [RNN cell] → h₃   (uses h₂ + "ate")
              ↓
"bone"→ [RNN cell] → h₄   (uses h₃ + "bone")
              ↓
         Output / Prediction
```

h is the hidden state — a compressed summary of everything the model has seen so far.

**The fatal problem — Vanishing Gradients:**

During training, the model adjusts its weights using a signal called a gradient that flows backwards through the network. In a long sentence, this signal gets weaker with each step backwards (multiplied by a small number repeatedly). By the time it reaches the first word, it has nearly vanished.

**Result:** In a long sentence, the model forgets what it read at the beginning. "The man who was standing near the big red bus that was parked outside the old church ___" — by the time you get here, the model has forgotten "man."

This is the **vanishing gradient problem**. LSTM was invented to fix it.

---

## 2. LSTM — Long Short-Term Memory

**Idea:** Add a second memory channel called the Cell State (C) — a "conveyor belt" that can carry information across very long sequences without degradation.

The LSTM has 4 gates that control what to remember and what to forget:

**Forget Gate:** "What old information should I delete from memory?"
If you're reading a new paragraph, you can forget the specific details of the last paragraph.

**Input Gate:** "What new information should I add to memory?"
The new paragraph's main topic should be added.

**Cell Update:** Actually performs the memory write.

**Output Gate:** "What should I output from the current memory state?"
Based on the memory so far, what's the right thing to say next?

```
     Cell State C  (long-term memory — the conveyor belt)
     ─────────────────────────────────────────────────────→
          ↑ forget        ↑ write new          ↑ read
     [FORGET Gate]  [INPUT Gate + Cell]  [OUTPUT Gate]
          ↑               ↑                    ↑
     ──────────────────────────────────────────────────
     Current word + previous hidden state hₜ₋₁
```

**Analogy:** Think of the LSTM as a person taking notes in a meeting.
Forget gate = erasing old notes that are no longer relevant.
Input gate = deciding whether to write down what was just said.
Cell state = the notebook itself — can hold info across the whole meeting.
Output gate = reading the right part of the notebook to answer a question.

**Where used:** Text generation, machine translation, speech recognition (before Transformers took over). Still used in time-series prediction.

---

## 3. GRU — Gated Recurrent Unit (2014)

GRU simplifies LSTM by merging the Forget and Input gates into a single **Update Gate**.

```
LSTM: Forget Gate + Input Gate + Cell State + Output Gate  (4 components)
GRU:  Update Gate + Reset Gate                             (2 components)
```

**Update Gate:** "How much of the old memory should I keep vs. replace?"

**Reset Gate:** "How much of the old memory should I use to compute the new state?"

**Tradeoffs:**
- GRU trains faster (fewer parameters)
- LSTM is often better for very long sequences
- In practice, results are similar enough that GRU is preferred when speed matters

**The honest truth:** Both LSTM and GRU have been largely replaced by Transformers for most NLP tasks. But they still appear in:
- Time-series prediction (stock prices, user activity)
- Streaming scenarios where you must process one token at a time
- Edge/mobile deployments where Transformers are too heavy

---

## 4. Bidirectional RNN/LSTM

Standard LSTM reads left to right. But to understand a word, you often need context from BOTH sides.

```
"The ____ was delicious"  → reading right-to-left helps: "delicious" suggests food
```

A Bidirectional LSTM runs TWO LSTMs simultaneously:
- One reads left → right
- One reads right → left
- Their hidden states are concatenated at each position

**Where used:** Named Entity Recognition, sentiment analysis where both sides of a sentence matter.

---

## 5. Seq2Seq (Encoder-Decoder with RNN)

**Problem:** Translating "The cat sat on the mat" (6 words in English) to French might produce a sentence of different length.

**Solution:** Use two RNNs.

```
ENCODER reads the entire input sentence → compresses it into a single context vector
DECODER takes that context vector → generates the output word by word

Input:   "The cat sat on the mat"
         ↓  ↓   ↓   ↓  ↓   ↓
      [ENCODER LSTM] → context vector c
                              ↓
                       [DECODER LSTM]
                        ↓   ↓   ↓
Output: "Le  chat  s'est  assis..."
```

**The bottleneck problem:** The entire meaning of a long sentence is compressed into ONE fixed-size vector. For long sentences, this vector can't hold everything. Information is lost.

**This bottleneck is exactly what Attention was invented to solve.**

---


---

# PART 4: ATTENTION MECHANISM

> The idea that transformed NLP. Worth understanding deeply.


---

## The Attention Problem Statement

In the Seq2Seq model, the decoder only gets one context vector — a single compressed summary of the entire input. For long sentences, this is too lossy.

**Attention's insight:** When generating each output word, let the decoder look back at ALL the encoder's hidden states — not just the final one — and decide which input words are most relevant RIGHT NOW.

**Analogy:** You're translating a document. For each word you're currently writing, you can glance back at the original document and highlight which words matter most for this specific output word. Attention is that highlighting mechanism.

---

## How Attention Works — Step by Step

**Translate: "The bank near the river" → French**

When generating the word for "bank" in French, the attention mechanism:

1. Looks at all 5 input word representations
2. Asks: "which of these inputs should I focus on?"
3. Assigns a weight (0 to 1) to each input word
4. Takes a weighted sum — words with high weights contribute more

```
Generating "banque" (French for bank):

Attention weights:
  "The"   → 0.05
  "bank"  → 0.45  ← the word itself
  "near"  → 0.05
  "the"   → 0.05
  "river" → 0.35  ← river is important context! (bank ≠ financial)

Weighted context = 0.45 × encode("bank") + 0.35 × encode("river") + ...
```

The model "attends" to "river" to correctly interpret "bank" as geographical, not financial.

---

## Scaled Dot-Product Attention — The Formula

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d ) · V
```

This looks scary but it's simple:

**Q = Query** — "What am I looking for?" The current word asks a question.

**K = Key** — "What do I have to offer?" Each word advertises itself.

**V = Value** — "What do I return if matched?" The actual content sent back.

**Analogy — Search engine:**
Q = your search query "best pizza NYC"
K = webpage titles (what each page is about)
V = the actual webpage content
Dot product = how well your query matches each title
Softmax = converts scores into probabilities (sum to 1)
Result = weighted mix of page contents, where the best matches contribute most

**Step by step:**
```
1. Q · Kᵀ         → similarity score between query and each key
2. / √d           → scale down to prevent very large numbers (d = vector size)
3. softmax(...)   → convert scores to weights that sum to 1
4. × V            → weighted sum of values = the attention output
```

---

## Multi-Head Attention

Instead of running attention once, run it multiple times in parallel with different learned Q, K, V matrices. Each "head" learns to attend to different types of relationships.

```
Head 1: "what's the grammatical subject of this verb?"
Head 2: "which pronoun does 'it' refer to?"
Head 3: "what's the emotional tone near this word?"
Head 4: "what's the named entity relationship here?"
```

Concatenate all head outputs → pass through a linear layer → final rich representation.

**Why it matters:** Language has many simultaneous dimensions of meaning. Multi-head attention lets the model capture all of them at once.

---

## Self-Attention

When Q, K, and V all come from the SAME sequence, it's called **self-attention**.

Each word attends to all other words in the same sentence to build its contextual representation.

```
Sentence: "The animal didn't cross the street because it was tired"

When encoding "it":
  Self-attention scores:
    "animal" → 0.72  ← high! "it" refers to the animal
    "street" → 0.08
    "tired"  → 0.15
    others   → low
```

The model learns that "it" refers to "animal" — not through rules, but through learned attention weights.

---

## Cross-Attention (Encoder-Decoder Attention)

In Seq2Seq with attention (and in Transformer decoders):
- Q comes from the decoder (the output being generated)
- K and V come from the encoder (the input that was read)

The decoder asks: "given where I am in generating the output, which parts of the input should I focus on?"

---

## Types of Attention — Quick Reference

| Type | Description | Used In |
|---|---|---|
| Bahdanau Attention | Original attention (2015), learned alignment | RNN-based translation |
| Luong Attention | Simpler dot-product variant | RNN-based systems |
| Self-Attention | Attends within same sequence | Transformers (BERT, GPT) |
| Cross-Attention | Attends across two sequences | Encoder-decoder Transformers |
| Multi-Head Attention | Multiple attention heads in parallel | All Transformer variants |
| Causal (Masked) Attention | Can only attend to past tokens | GPT (autoregressive) |
| Sparse Attention | Attends to subset of tokens | Long documents, Big Bird |
| Flash Attention | Same math, faster GPU computation | Modern LLMs (speed optimization) |

---

## Flash Attention — Why It Matters Now

Standard attention requires loading Q, K, V matrices into slow GPU memory repeatedly. For long sequences, this is the bottleneck.

Flash Attention (2022) restructures the computation to stay in fast GPU cache — producing the exact same mathematical result, just 2–4× faster with much less memory.

**Why you should know this:** Flash Attention is why modern LLMs can handle 100K+ token context windows efficiently. Without it, GPT-4 handling a 128K context would be impractically slow.

---


---

# PART 5: THE TRANSFORMER ARCHITECTURE

> The architecture that powers BERT, GPT, T5, and every modern LLM.
> Paper: "Attention Is All You Need" (Vaswani et al., 2017, Google)

**The key insight:** You don't need RNNs at all. Attention alone is sufficient — and much faster because it's parallelisable.

---

## Transformer vs RNN — The Core Difference

| Property | RNN/LSTM | Transformer |
|---|---|---|
| Processes words | One at a time (sequential) | All at once (parallel) |
| Memory of long sequences | Struggles (vanishing gradient) | Strong (direct attention to any word) |
| Training speed | Slow | Fast (GPU-parallelisable) |
| Long-range dependencies | Weak | Strong |

---

## The Transformer Architecture — Layer by Layer

```
INPUT SIDE (Encoder)                    OUTPUT SIDE (Decoder)
─────────────────────────               ──────────────────────────
Input Embedding                         Output Embedding
    +                                       +
Positional Encoding                     Positional Encoding
    ↓                                       ↓
Multi-Head Self-Attention              Masked Multi-Head Self-Attention
    ↓                                       ↓
Add & Layer Norm                       Add & Layer Norm
    ↓                                       ↓
Feed-Forward Network              ←── Cross-Attention (Q from decoder,
    ↓                                       K,V from encoder)
Add & Layer Norm                       Add & Layer Norm
    ↓                                       ↓
   × N (stacked)                       Feed-Forward Network
    ↓                                       ↓
Encoder Output                         Add & Layer Norm
                                           ↓
                                        × N (stacked)
                                           ↓
                                       Linear + Softmax
                                           ↓
                                       Output Token
```

---

## Each Component Explained

**Input Embedding:** Convert each token (word/subword) into a dense vector (typically 512 or 768 dimensions in BERT).

**Positional Encoding:** Transformers process all words at once — so they have no inherent sense of word order. Positional encoding adds a unique number pattern to each position's embedding so the model knows word 1 is before word 2.

```
Position 1: embedding + [sin(1/1), cos(1/1), sin(1/100), ...]
Position 2: embedding + [sin(2/1), cos(2/1), sin(2/100), ...]
```

Uses sine and cosine functions at different frequencies — so each position gets a unique, learnable signature.

**Feed-Forward Network:** After attention mixes information across words, a simple two-layer neural net processes each word's representation independently. This is where the "thinking" happens after "listening."

**Add & Layer Norm:** Add the input to the output of each sublayer (residual connection — prevents gradient from vanishing) and normalise to keep activations stable.

**Masked Self-Attention (Decoder only):** When the decoder generates word 4, it cannot see words 5, 6, 7 (those are future — they don't exist yet). The mask sets those attention weights to −∞ before softmax, making them effectively zero.

**Cross-Attention (Decoder only):** The decoder queries the encoder's output. Q from decoder = "what am I trying to generate now?" K,V from encoder = "what did the input say?"

---

## Positional Encoding Deep Dive

**Why not just use position numbers [1, 2, 3...]?**

The model would struggle to generalise to sequences longer than those seen in training.

**Why sine/cosine?**

It creates a smooth, continuous representation. Every position gets a unique pattern. The relative distance between positions can be computed mathematically from the encodings alone. Allows the model to generalise to longer sequences.

---


---

# PART 6: BERT, GPT, AND THE MODERN LLM FAMILY

---

## BERT (2018, Google)

**Full name:** Bidirectional Encoder Representations from Transformers

**Architecture:** Transformer ENCODER only (no decoder).

**Direction:** Bidirectional — reads the whole sentence at once, left AND right context simultaneously.

**Training tasks:**

MLM (Masked Language Model): randomly mask 15% of tokens and predict them.
```
"The [MASK] sat on the mat" → predict "cat"
```

NSP (Next Sentence Prediction): given two sentences, predict if B follows A.
```
A: "The cat sat on the mat"   B: "The mat was red"  → IsNext: True
A: "The cat sat on the mat"   B: "Stock prices fell" → IsNext: False
```

**BERT sizes:**
- BERT-Base: 12 layers, 110M parameters
- BERT-Large: 24 layers, 340M parameters

**Best for (understanding/classification):**
- Sentiment analysis
- Named Entity Recognition
- Question answering (finds answer span in a passage)
- Semantic similarity

**Not good for:** Text generation. BERT is an encoder — it understands but doesn't generate.

---

## GPT Family (2018–now, OpenAI)

**Full name:** Generative Pre-trained Transformer

**Architecture:** Transformer DECODER only (no encoder).

**Direction:** Left-to-right (causal/autoregressive). Each word can only attend to previous words.

**Training task:**

CLM (Causal Language Model): predict the next word given all previous words.
```
"The cat sat on the" → predict "mat"
```

**GPT evolution:**

GPT-1 (2018): 117M parameters. Proved the pre-train/fine-tune approach works.
GPT-2 (2019): 1.5B parameters. So good at generation OpenAI initially withheld it.
GPT-3 (2020): 175B parameters. Few-shot learning — works on tasks it wasn't trained for.
InstructGPT (2022): GPT-3 + RLHF. Follows instructions, safer, more useful.
ChatGPT (2022): InstructGPT with conversational interface. 100M users in 2 months.
GPT-4 (2023): Multimodal (text + images), 1T+ parameters (estimated).

**Best for (generation tasks):**
- Chat and conversational AI
- Content generation, copywriting
- Code generation
- Summarisation, translation

---

## T5 (2019, Google)

**Full name:** Text-To-Text Transfer Transformer

**Architecture:** Full Encoder + Decoder.

**The elegant idea:** Frame EVERY NLP task as text-in → text-out.

```
"translate English to French: The cat sat on the mat"
  → "Le chat s'est assis sur le tapis"

"sentiment: I love this product"
  → "positive"

"summarize: [long article]"
  → "[short summary]"
```

One unified model handles all tasks. Just prefix the input with the task name.

---

## Other Important Models

**RoBERTa (2019, Meta):** BERT trained longer on more data, with the NSP task removed. Consistently outperforms BERT on benchmarks.

**DistilBERT (2019, Hugging Face):** BERT distilled to 66% of size, retaining 97% of performance. Trained using knowledge distillation — a small student model learns to mimic the outputs of the large teacher model. Used when speed or mobile deployment matters.

**ALBERT (2019, Google):** BERT with two parameter reduction techniques: shared weights across layers, and factorised embedding parameterisation. Much smaller without big accuracy loss.

**XLNet (2019, CMU + Google):** Addresses BERT's limitation that masked tokens don't see each other during training. Uses permutation language modelling — trains on all possible word orderings.

**DeBERTa (2020, Microsoft):** Adds disentangled attention — separate attention for word content and word position. Current state-of-the-art on many NLP benchmarks.

**mBERT and XLM-R:** Multilingual BERT variants. Trained on 100+ languages. LinkedIn uses this for multilingual content understanding.

---

## LLaMA and Open-Source LLMs (2023+)

**LLaMA (Meta, 2023):** High-quality open-source LLM. Triggered an explosion of fine-tuned variants (Alpaca, Vicuna, WizardLM). Proved you don't need GPT-4 scale for many tasks.

**Mistral 7B (2023):** 7B parameter model outperforming LLaMA 13B. Uses Grouped Query Attention (GQA) and Sliding Window Attention for efficiency.

**Grouped Query Attention (GQA):** Instead of one K,V per attention head, group multiple heads to share K,V. Reduces memory without much accuracy loss. Used in LLaMA 2/3, Mistral.

**Sliding Window Attention:** Instead of attending to all previous tokens, each token only attends to the nearest W tokens. Reduces attention from O(n²) to O(n×W). Enables very long contexts efficiently.

---

## Quick Model Comparison Table

| Model | Architecture | Direction | Best For | Size |
|---|---|---|---|---|
| BERT | Encoder only | Bidirectional | Understanding, Classification | 110M–340M |
| RoBERTa | Encoder only | Bidirectional | Same as BERT, better | 125M–355M |
| DistilBERT | Encoder only | Bidirectional | Fast/mobile classification | 66M |
| GPT-3/4 | Decoder only | Left-to-right | Generation, chat | 175B–1T+ |
| T5 | Encoder+Decoder | Both | Any task, text-to-text | 60M–11B |
| LLaMA 3 | Decoder only | Left-to-right | Open-source generation | 8B–70B |
| Mistral 7B | Decoder only | Left-to-right | Efficient generation | 7B |

---

---

# PART 7: CNN FOR NLP + FINE-TUNING + RLHF


---

## CNN for NLP

CNNs (Convolutional Neural Networks) were originally built for images — a filter slides across pixels detecting patterns like edges or shapes. In NLP, the filter slides across words detecting linguistic patterns.

**How it works:**

```
Sentence: ["I", "love", "this", "app", "so", "much"]
Filter size = 2 (looks at 2 words at a time)

Window 1: ["I", "love"]        → activation score: 0.2
Window 2: ["love", "this"]     → activation score: 0.7
Window 3: ["this", "app"]      → activation score: 0.9  ← strong positive signal
Window 4: ["app", "so"]        → activation score: 0.4
Window 5: ["so", "much"]       → activation score: 0.6

Max pooling → takes the highest score (0.9) as the feature
```

Multiple filters of sizes 2, 3, and 4 run in parallel, detecting bigrams, trigrams, and 4-grams simultaneously.

**Where CNNs shine in NLP:**
- Fast text classification (spam detection, intent classification)
- Detecting specific phrases or patterns
- Sentence-level sentiment analysis

**Where CNNs fail:**
- Long-range dependencies (can't link beginning to end of a long sentence)
- Understanding overall sentence structure

---

## Transfer Learning and Fine-Tuning

**Pre-training:** Train a large model on a huge general corpus (Wikipedia, books, Common Crawl). This costs millions of dollars and weeks of compute. The model learns general language representations.

**Fine-tuning:** Take that pre-trained model. Continue training on YOUR small, task-specific dataset for a few epochs. The model adapts its general knowledge to your specific task.

```
Pre-trained BERT (general language knowledge)
    ↓
Fine-tune on 10,000 LinkedIn job descriptions
    ↓
Model that classifies job seniority (Junior/Mid/Senior/Lead)
```

**Why this works:** Pre-training teaches the model language. Fine-tuning teaches the model your specific task. You're not starting from scratch — you're redirecting existing knowledge.

**Few-shot learning:** Instead of fine-tuning, just give GPT-4 a few examples in the prompt. It learns the pattern in context without any weight updates.

```
Prompt: "Classify sentiment:
  'I love this product' → positive
  'Worst app ever' → negative
  'This LinkedIn feature is amazing' → "
```

GPT-4 answers: "positive" — without any training.

---

## RLHF — Reinforcement Learning from Human Feedback

This is how ChatGPT became ChatGPT. A standard GPT model trained on next-word prediction is good at generating text — but not necessarily helpful, safe, or aligned with what humans want.

**The 3-step process:**

**Step 1: Supervised Fine-Tuning (SFT)**

Human trainers write ideal example conversations — the behaviour you want. Fine-tune the model on these examples to create a baseline well-behaved model.

**Step 2: Train a Reward Model**

Have the fine-tuned model generate multiple responses to the same prompt. Human labellers rank these responses: Response A > Response B > Response C.

Train a separate "Reward Model" to predict these rankings — it learns to score responses 0–10 based on how humans would rate them.

**Step 3: PPO Reinforcement Learning**

Use the Reward Model as the environment's reward signal. Run PPO (Proximal Policy Optimization — an RL algorithm) to adjust the main model's weights.

Good response (reward model gives high score) → reinforce those weights.
Bad response (reward model gives low score) → reduce probability of those outputs.

```
Prompt → LLM generates response → Reward Model scores it → PPO updates LLM
                 ↑_____________________________________________↓ (repeat thousands of times)
```

**Why PPO specifically?** It's a "conservative" RL algorithm — it constrains how much the model changes per update, preventing the model from "reward hacking" (finding ways to get high scores that aren't actually helpful).

**DPO — Direct Preference Optimization (2023):** A simpler, more stable alternative to RLHF. Skips the reward model entirely — directly trains the LLM on human preference pairs (A preferred over B). Produces similar results with less complexity. Used in LLaMA 2, Mistral, and Claude.

---

## RAG — Retrieval-Augmented Generation

Not a training technique — an inference technique. But you need to know it.

**Problem:** LLMs hallucinate. They can't access real-time information. They have a knowledge cutoff.

**Solution:** At inference time, retrieve relevant documents from a knowledge base and inject them into the prompt.

```
User asks: "What were LinkedIn's Q3 results?"

RAG system:
  1. Encode question as a vector
  2. Search vector database for similar documents
  3. Retrieve top 3 relevant paragraphs
  4. Inject into prompt: "Use this context: [paragraphs] Answer: [question]"
  5. LLM generates grounded answer

User gets: factual, cited answer instead of hallucination
```

**Where used:** Enterprise chatbots, internal knowledge bases, customer support AI, LinkedIn's Help Center.

---

---

# PART 8: EVALUATION METRICS

> The metrics you'll be asked about in interviews. Know what each measures and when to use it.

---

## Classification Metrics

**Accuracy:** Fraction of predictions that are correct. Misleading when classes are imbalanced (if 95% of emails are not spam, a model that says "not spam" always is 95% accurate but useless).

**Precision:** Of all the things the model labelled positive, how many actually were?
```
Precision = True Positives / (True Positives + False Positives)
```
High precision = few false alarms. Matters when false positives are costly (e.g. marking real emails as spam).

**Recall:** Of all the actual positives, how many did the model find?
```
Recall = True Positives / (True Positives + False Negatives)
```
High recall = few misses. Matters when missing a case is costly (e.g. cancer detection, fraud detection).

**F1 Score:** Harmonic mean of Precision and Recall. Best single metric when you need to balance both.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to use which:**

Spam filter: High Precision (don't want real emails in spam) over High Recall.
Fraud detection: High Recall (catch every fraud) over High Precision.
Medical diagnosis: High Recall (don't miss a sick patient).

---

## Language Generation Metrics

**BLEU (Bilingual Evaluation Understudy):** Measures how much n-gram overlap exists between the model's output and a reference translation.

```
Reference: "The cat sat on the mat"
Output:    "The cat is sitting on the mat"
BLEU checks: how many 1-grams, 2-grams, 3-grams, 4-grams match?
```

Scale: 0 to 1 (higher is better). BLEU > 0.4 is generally good for translation.

Limitation: only checks exact n-gram matches. "Good" and "excellent" score zero overlap even though they mean the same thing.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Used for summarisation. Measures overlap between generated summary and reference summary. More recall-focused than BLEU.

**Perplexity:** How "surprised" a language model is by a text. Lower perplexity = the model finds the text predictable = better language model.

```
Perplexity = 10^(average cross-entropy loss)
```

A perplexity of 20 means the model is as confused as if it had to choose uniformly among 20 words at each step.

**BERTScore:** Uses BERT embeddings to measure semantic similarity between output and reference — much better than BLEU at capturing meaning even when exact words differ.

---

## Information Retrieval Metrics (Search + Recommendation)

**Precision@K:** Of the top K results returned, how many are relevant?
```
Precision@5 = (relevant items in top 5) / 5
```

**Recall@K:** Of all relevant items that exist, how many appear in the top K?

**NDCG (Normalized Discounted Cumulative Gain):** Measures ranking quality. A relevant result at position 1 is worth more than at position 5.

```
DCG@5 = rel₁ + rel₂/log₂(2) + rel₃/log₂(3) + rel₄/log₂(4) + rel₅/log₂(5)
NDCG = DCG / ideal DCG (if perfect ranking)
```

Used in: LinkedIn job recommendations, search ranking, content feed ranking.

**MRR (Mean Reciprocal Rank):** Average of (1/rank of first relevant result) across all queries.
```
Query 1: first relevant result at position 2 → 1/2 = 0.5
Query 2: first relevant result at position 1 → 1/1 = 1.0
MRR = (0.5 + 1.0) / 2 = 0.75
```

---

## LLM-Specific Metrics

**Hallucination Rate:** Percentage of outputs containing factually incorrect statements. Hard to measure automatically — often requires human evaluation or a separate fact-checking model.

**Toxicity Score:** Using a classifier (e.g. Perspective API) to measure harmful content in outputs. Critical for content safety.

**Human Preference (Win Rate):** Show two model outputs to a human: which is better? The model preferred more often "wins." Used in RLHF evaluation and model comparisons.

---

---

# PART 9: NLP IN PRACTICE 

---

## Scenario 1: LinkedIn's Job Search

**Stack:**

1. Query understanding: BERT-based model classifies intent (job title? skill? company?) and expands synonyms ("SWE" → "Software Engineer").

2. Retrieval: Dual-encoder model (Two-Tower) converts query and job descriptions into vectors. ANN search finds nearest job vectors.

3. Re-ranking: Cross-encoder model re-scores top 500 results with full attention between query and job description (more expensive but more accurate than the dual encoder).

4. Personalisation: User's profile, skills, and past behaviour modulate the final ranking.

**Decision point:** Dual encoder (fast, approximate) vs cross-encoder (slow, precise). You can't run a cross-encoder on all 50 million jobs. So you use dual encoder to get to 500, then cross-encoder to rank those 500. This is the retrieve-then-rerank pattern.

---

## Scenario 2: LinkedIn's Post Quality Filter

**Task:** Binary classification — is this post high quality or spam?

**Stack:**

1. BERT-based classifier fine-tuned on LinkedIn posts labelled by human raters.
2. Confidence threshold: posts above 0.9 confidence are auto-approved or auto-rejected. Posts between 0.6–0.9 go to human review.
3. Continual learning: human review decisions feed back as new training data.

**Decision point:** Where to set the threshold. Low threshold → more automation, higher false positives (good content rejected). High threshold → less automation, more human review cost. This is a business decision that must balance quality, cost, and creator experience.

---

## Scenario 3: LinkedIn's AI Writing Assistant

**Task:** Given partial post, suggest completion or rephrase.

**Stack:** Fine-tuned GPT-style model + RLHF to ensure professional tone.

**RAG component:** Inject user's profile (skills, job, industry) and trending topics into context to personalise suggestions.

**Metric:** Not BLEU — human preference rate. A/B test: do users who see AI suggestions post more, and do those posts get more engagement?

---

## Mental Model for NLP Decisions

```
CHOOSING THE RIGHT MODEL:

Is this a classification / understanding task?
  → BERT family (RoBERTa, DistilBERT)

Is this a generation task?
  → GPT family, LLaMA, Claude

Do you need seq-to-seq (translation, summarisation)?
  → T5, mBART, BART

Is latency critical (mobile, real-time)?
  → DistilBERT, TinyBERT, quantised models

Do you need to ground answers in documents (no hallucination)?
  → RAG architecture

Do you want the model to follow instructions safely?
  → RLHF fine-tuned model (InstructGPT, Claude, LLaMA-chat)
```

---


---

# PART 10: QUICK REFERENCE CHEAT SHEET

---

## All Abbreviations

| Abbreviation | Full Name | One-Line Meaning |
|---|---|---|
| NLP | Natural Language Processing | Getting computers to understand language |
| BoW | Bag of Words | Count word frequencies, ignore order |
| TF-IDF | Term Freq–Inverse Document Freq | Weight words by how distinctive they are |
| Word2Vec | Word to Vector | Static word embeddings from context prediction |
| GloVe | Global Vectors | Static embeddings from co-occurrence statistics |
| RNN | Recurrent Neural Network | Processes sequences one step at a time |
| LSTM | Long Short-Term Memory | RNN with gated memory to avoid forgetting |
| GRU | Gated Recurrent Unit | Simpler, faster LSTM variant |
| Seq2Seq | Sequence to Sequence | Encoder-decoder for translation/summarisation |
| BERT | Bidirectional Encoder Reps from Transformers | Bidirectional text understanding model |
| GPT | Generative Pre-trained Transformer | Left-to-right text generation model |
| MLM | Masked Language Model | BERT's pretraining task (predict hidden words) |
| CLM | Causal Language Model | GPT's pretraining task (predict next word) |
| NSP | Next Sentence Prediction | BERT pretraining task |
| T5 | Text-to-Text Transfer Transformer | All NLP tasks as text-in text-out |
| SFT | Supervised Fine-Tuning | Fine-tune on human-written examples |
| RLHF | Reinforcement Learning from Human Feedback | Train from human preference rankings |
| DPO | Direct Preference Optimization | Simpler RLHF alternative |
| PPO | Proximal Policy Optimization | RL algorithm used in RLHF |
| RAG | Retrieval-Augmented Generation | Ground LLM answers in retrieved documents |
| QA | Question Answering | NLP task of answering questions |
| NER | Named Entity Recognition | Extracting names, places, dates from text |
| BLEU | Bilingual Evaluation Understudy | n-gram overlap metric for translation |
| ROUGE | Recall-Oriented Understudy for Gisting | n-gram overlap metric for summarisation |
| NDCG | Normalised Discounted Cumulative Gain | Ranking quality metric |
| MRR | Mean Reciprocal Rank | Average rank of first relevant result |
| ANN | Approximate Nearest Neighbor | Fast vector similarity search |
| Q | Query | "What am I looking for?" in attention |
| K | Key | "What do I have?" in attention |
| V | Value | "What do I return?" in attention |
| GQA | Grouped Query Attention | Efficient attention in LLaMA/Mistral |
| LoRA | Low-Rank Adaptation | Efficient fine-tuning by adding small matrices |
| KV Cache | Key-Value Cache | Store past attention keys/values to speed up generation |

---

## Model Architecture One-Liners

```
One-Hot     → [1,0,0,0]  — just an ID, no meaning
Word2Vec    → [0.82, 0.14, 0.65]  — meaning, static
BERT        → Encoder. Bidirectional. Understands.
GPT         → Decoder. Left-to-right. Generates.
T5          → Encoder+Decoder. All tasks as text-in/text-out.
RNN         → Sequential. Forgets early words.
LSTM        → RNN + memory gates. Remembers longer.
GRU         → Simplified LSTM. Faster.
Transformer → All words at once. Attention everywhere. Parallelisable.
RAG         → LLM + search. Grounds answers in real documents.
RLHF        → Human preference → reward model → RL fine-tuning.
```

---

## Interview Answer Templates

**"Explain attention in one minute:"**

Attention lets a model focus on the most relevant parts of the input when processing each output. When translating "bank" near "river," the model attends more strongly to "river" to understand it's geographical, not financial. Mathematically, it computes a weighted sum where each input word's weight comes from how similar it is (via dot product) to the current output position.

**"Why BERT over Word2Vec?"**

Word2Vec gives each word one fixed vector regardless of context — so "bank" always has the same vector whether near "river" or "loan." BERT reads the whole sentence bidirectionally and generates a different vector for each word depending on its full context. This contextual understanding makes BERT dramatically better for understanding tasks.

**"Why GPT can't use BERT's architecture:"**

BERT is bidirectional — it reads the whole sentence at once, which is perfect for understanding but impossible for generation. Generation requires producing one word at a time, where each word can only depend on previous words (you haven't produced future words yet). GPT uses a decoder-only architecture with causal masking — each position can only attend to positions before it. This makes it autoregressive and capable of generation.

**"What is RLHF and why does it matter:"**

Pre-training on text makes a model good at predicting next words — but not necessarily helpful or safe. RLHF aligns the model to human preferences: (1) fine-tune on human-written examples, (2) train a reward model on human rankings of outputs, (3) use RL to maximise the reward model score. This transformed GPT-3 (impressive but erratic) into ChatGPT (helpful, safe, instruction-following).

---

