# Protein-Specific-Drug-Generation

This project is inspired by [this paper](https://www.nature.com/articles/s41598-020-79682-4), which implements a transformer neural network for protein-specific drug generation. Based on a amino acid sequence of a protein target, the goal is to find the SMILES representation of a suitable drug. The goal of this project is to compare the performance of different model architectures on this task. The evaluated models are those capable of understanding sequence data, such as transformer structures like in the referenced paper, trained from scratch or by fine-tuning existing LLMs, as well as LSTM and RNN models.

## Data processing
The data used is the same as in the referenced paper: BindingDB, which contains measured binding affinities of interactions between proteins and drug-like molecules. The proteins were filtered to ensure no missing values, and to include only those from humans, rats, mice, or tauruses, and of medium size. Each element of the amino acid sequence and SMILES sequence is then encoded.

## Models and results

Different model architectures capable of understanding sequence-like data were trained and compared on a test set. The models were trained on the full dataset of 337,459 protein–SMILES pairs from BindingDB.

### Transformer (trained from scratch)

An encoder-decoder transformer with the following architecture was trained from scratch:

| Hyperparameter | Value |
|---|---|
| `d_model` | 128 |
| `num_heads` | 2 |
| `num_encoder_layers` | 2 |
| `num_decoder_layers` | 2 |
| `d_ff` | 256 |
| `dropout` | 0.1 |
| Parameters | 687,295 |
| Optimizer | Adam (lr=1e-4) |
| Batch size | 16 (dynamic padding) |

Training used **dynamic per-batch padding** (each batch padded to its own max length) rather than global padding, which reduced peak memory from ~6 GB to ~2 GB and allowed training on the full dataset on a single T4 GPU. A critical fix over earlier implementations was the **shifted-target objective**: the decoder input is `tgt[:-1]` (SOS..t_{L−1}) and the loss target is `tgt[1:]` (t_1..EOS). Without this shift, causal self-attention allows the model to trivially copy each input token, producing zero autoregressive accuracy regardless of loss.

Training ran for **5 epochs** (~3h 15min on a T4 GPU, 105,455 total steps).

**Training loss curve:**

| Epoch | Avg Loss | TF Accuracy |
|---|---|---|
| 1 | 1.145 | 66% |
| 2 | 0.937 | 71% |
| 3 | 0.869 | 74% |
| 4 | 0.824 | 74% |
| 5 | 0.793 | 75% |

**Held-out evaluation (greedy decoding):**

| Metric | Validation | Test |
|---|---|---|
| TF loss | 0.872 | 0.829 |
| TF accuracy | 69.8% | 70.9% |
| SMILES validity (RDKit) | 46.2% | 57.0% |
| Exact match | 0% | 0% |

SMILES validity measures the fraction of generated sequences that parse as chemically valid molecules according to RDKit. The ground-truth sequences are 100% valid. Exact match is 0% as expected for this open-ended generation task.

**Qualitative observations:** The model generates drug-like scaffolds such as biaryl amides, piperidine/piperazine-containing compounds, and heteroaromatic kinase inhibitor-like structures. Remaining failure modes include occasional degenerate outputs (long alkane chains) and mode collapse under greedy decoding (the same scaffold generated for multiple different protein inputs).

The training experiment is tracked at: https://wandb.ai/oriane-cavrois/protein-drug-gen

### ProtBERT (encoder fine-tuning)

[ProtBert](https://huggingface.co/Rostlab/prot_bert) from the HuggingFace library was used as the encoder. It is specifically pre-trained on amino acid sequences in a self-supervised fashion (masked token prediction). An attention-based decoder was added; in the first iteration only the decoder weights were trained while the ProtBERT encoder was frozen (420M frozen parameters).

**Architecture (v2):**

| Hyperparameter | Value |
|---|---|
| Encoder | ProtBERT (frozen) |
| `d_model` | 1024 (matches ProtBERT output) |
| `num_heads` | 4 |
| `num_decoder_layers` | 2 |
| `d_ff` | 512 |
| `dropout` | 0.1 |
| Trainable params | ~6.3M (decoder only) |
| Optimizer | Adam (lr=1e-3) |
| Batch size | 64 (dynamic padding) |
| Subsample | 1/12 of data (~28K pairs) |

Two critical fixes were applied over an initial v1 implementation: (1) the **shifted-target objective** (same as the scratch transformer — decoder input `tgt[:-1]`, loss target `tgt[1:]`); (2) **proper ProtBERT tokenization** — protein sequences are re-encoded per batch using `BertTokenizer` with space-separated amino acids, rather than the custom symb2index vocabulary. A key architectural fix was increasing `d_ff` from 48 to 512 — the original 1024→48→1024 bottleneck caused complete mode collapse regardless of other settings.

Training ran for **5 epochs** (~75 min/epoch on a T4 GPU, ~6.25h total, 439 batches/epoch).

**Training loss curve:**

| Epoch | Avg Loss |
|---|---|
| 1 | 1.527 |
| 2 | ~1.26 |
| 3 | 1.265 |
| 4 | ~1.253 |
| 5 | **1.253** |

V1 (d_ff=48) plateau: ~2.54 across all epochs. The d_ff fix alone yielded a **~51% loss reduction**.

The training experiment is tracked at: https://wandb.ai/oriane-cavrois/Protein-Specific-Drug-Generation

**Qualitative inference results (greedy decoding, 5 test examples):**

Mode collapse was fully resolved — the model generates diverse SMILES with valid drug-like fragments (e.g. `CC(C)(C)`, `CC(=O)N1CC`, `CC(C)C[C@H]1`). However, greedy decoding causes repetitive carbon-chain runoff after ~20 tokens. This is a known decoding artifact: the most probable continuation at each step drifts toward simple carbon chains. Temperature or nucleus (top-p) sampling is expected to resolve this.

| Metric | Value |
|---|---|
| Mode collapse | None (resolved) |
| Char overlap (greedy) | 10–24% |
| Drug-like prefix generation | Yes |

---

## Potential next steps

- **Sampling strategies**: Replace greedy decoding with temperature or nucleus (top-p) sampling to address mode collapse and improve output diversity.
- **SMILES validity filtering**: Post-process generated sequences with RDKit and re-sample invalid ones.
- **Drug-likeness metrics**: Evaluate generated molecules on QED (quantitative estimate of drug-likeness) and SA score (synthetic accessibility), not just SMILES validity.
- **More training**: The val loss (0.872) is still well above 0.5, and the model shows no sign of overfitting (train–val gap is small). Additional epochs or a larger architecture (`d_model=256`, more layers) would likely improve both validity and diversity.
- **ProtBERT fine-tuning**: Unfreeze the ProtBERT encoder and fine-tune end-to-end, which should produce better protein representations and improve conditioning.
- **Binding affinity evaluation**: Use docking software (e.g. AutoDock Vina) or a pre-trained affinity predictor to assess whether generated molecules are actually predicted to bind the input protein.
