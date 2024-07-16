# Protein-Specific-Drug-Generation

This project is inspired by [this paper](https://www.nature.com/articles/s41598-020-79682-4), which implements a transformer neural network for protein-specific drug generation. Based on a amino acid sequence of a protein target, the goal is to find the SMILES representation of a suitable drug. The goal of this project is to compare the performance of different model architectures on this task. The evaluated models are those capable of understanding sequence data, such as transformer structures like in the referenced paper, trained from scratch or by fine-tuning existing LLMs, as well as LSTM and RNN models.

## Data processing
The data used is the same as in the referenced paper: BindingDB, which contains measured binding affinities of interactions between proteins and drug-like molecules. The proteins were filtered to ensure no missing values, and to include only those from humans, rats, mice, or tauruses, and of medium size. Each element of the amino acid sequence and SMILES sequence is then encoded.

## Models and results
Different model architectures capable of understanding sequence-like data were trained and compared on a test set. The models were trained on a sample size of approximately 169k samples, and their performance was tested on a sample of approximately 26k records.
For existing LLMs, my first choice was to use [ProtBert](https://huggingface.co/Rostlab/prot_bert) from the HuggingFace library, as it is specifically trained on amino acid sequences. This model was trained in a self-supervised fashion (no labeling), by inferring one masked amino acid in the sequence. To use it as a generative model, I created an encoder-decoder model where ProtBert is used as the encoder, and an attention-based decoder, where the hidden state from the encoder is used as the input for the decoder. In the first iteration, the weights of the encoder (ProtBert) remained unchanged, and only the weights of the decoder were learned.

Both models were trained for only 2 epochs, as the dataset size is substantial, and this was sufficient to achieve promising results.
To evaluate the models, each element of each sequence is compared to assess the accuracy of the models.

**Summary of the models performances:**
| Performance metrics  | Transformer | Fine tuned ProtBert  |
| ------------- | ------------- | ------------- |
| Accuracy  | 0.99999915  | 0.99999872  |
| F1 score  | 0.99999915  | 0.99999881  |

The models seem to learn very quickly and reach very high accuracy on the test set. The inferred sequence would need to be further analyzed though, to study the chemical feasibility, the binding affinity (taking into account technical limitations) and the toxicity.
