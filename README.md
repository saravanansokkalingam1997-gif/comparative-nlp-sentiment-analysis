# 🎬 Comparative NLP Sentiment Analysis: LSTM, ULMFiT & BERT

**IMDb Movie Review Sentiment Classification using Deep Learning, Transfer Learning, and Transformer Architecture**

This project presents an end-to-end comparison of three deep learning approaches for binary sentiment classification on the IMDb movie review dataset:

* **Custom LSTM** — manually implemented from scratch
* **ULMFiT** — AWD-LSTM with language-model pretraining and transfer learning
* **BERT** — pretrained Transformer with fine-tuning

The project demonstrates the progression from a traditional recurrent neural network trained from scratch to modern pretrained language models.

---

## 📈 Results at a Glance

| Model           | Architecture                    | Pre-training | Test Accuracy | Precision | Recall | F1 Score |
| :-------------- | :------------------------------ | :----------: | :-----------: | :-------: | :----: | :------: |
| **Custom LSTM** | Manual LSTM                     |       ❌      |   **84.28%**  |   0.8375  | 0.8505 |  0.8440  |
| **ULMFiT**      | AWD-LSTM + Transfer Learning    |       ✅      |   **91.84%**  |   0.9467  | 0.8866 |  0.9157  |
| **BERT**        | `bert-base-uncased` Transformer |       ✅      |   **93.42%**  |   0.9309  | 0.9381 |  0.9345  |

### Key Result

The experiments show a clear performance improvement as increasingly powerful pretrained language representations are introduced:

**Custom LSTM → ULMFiT → BERT**

The Custom LSTM achieved **84.28%** test accuracy, while ULMFiT improved performance to **91.84%** through language-model pretraining and transfer learning. BERT achieved the highest performance with **93.42%** test accuracy.

---

## 📂 Project Structure

```text
nlp-sentiment-analysis/
│
├── README.md
├── requirements.txt
│
└── notebooks/
    ├── 01_Custom_LSTM.ipynb
    ├── 02_ULMFiT_AWD_LSTM.ipynb
    └── 03_BERT.ipynb
```

---

## 🧠 Models

### 1. Custom LSTM

The first model is a manually implemented LSTM designed to understand the fundamentals of recurrent neural networks without relying on a pretrained language model.

#### Architecture

```text
Input Text
    ↓
Tokenization
    ↓
Vocabulary
    ↓
Embedding
    ↓
Manual LSTM Gates
    ├── Forget Gate
    ├── Input Gate
    ├── Candidate State
    └── Output Gate
    ↓
Dropout
    ↓
Linear Classifier
    ↓
Sentiment Prediction
```

#### Key Techniques

* PyTorch implementation
* Manually implemented LSTM gates
* Word-level vocabulary
* Padding handling
* Padding masking using `torch.where`
* Gradient clipping
* Adam optimizer
* `ReduceLROnPlateau`
* Early stopping
* Best-model checkpointing

#### Result

**Test Accuracy: 84.28%**

---

### 2. ULMFiT — AWD-LSTM

The second approach uses **ULMFiT (Universal Language Model Fine-tuning)** with the AWD-LSTM architecture.

Instead of training the language representations entirely from scratch for sentiment classification, the model first learns general language patterns through language-model pretraining and then transfers those representations to the classification task.

#### Training Pipeline

```text
IMDb Text
    ↓
Language Model Pretraining
    ↓
AWD-LSTM Encoder
    ↓
Save Pretrained Encoder
    ↓
Transfer Encoder to Classifier
    ↓
Initial Classifier Training
    ↓
Progressive Unfreezing
    ↓
Discriminative Learning Rates
    ↓
Sentiment Classification
```

#### Key Techniques

* FastAI
* AWD-LSTM
* Language-model pretraining
* Encoder transfer learning
* Progressive layer unfreezing
* Discriminative learning rates
* Vocabulary alignment
* One Cycle learning-rate policy
* Validation and test evaluation

#### Result

**Test Accuracy: 91.84%**

---

### 3. BERT

The final approach uses **BERT (`bert-base-uncased`)**, a pretrained Transformer model.

BERT provides bidirectional contextual representations, allowing the model to consider information from both directions when interpreting a sentence.

#### Fine-Tuning Strategy

```text
IMDb Review
    ↓
BERT Tokenizer
    ↓
Pretrained BERT
    ↓
Classification Head
    ↓
Initial Head Training
    ↓
Full Model Fine-Tuning
    ↓
Sentiment Prediction
```

#### Key Techniques

* Hugging Face Transformers
* `bert-base-uncased`
* Pretrained Transformer representations
* BERT tokenizer
* Two-stage fine-tuning
* Frozen-base training
* Full-model fine-tuning
* Maximum sequence length of 384 tokens

#### Result

**Test Accuracy: 93.42%**

---

## 📊 Dataset

This project uses the **Stanford IMDb Movie Review Dataset** for binary sentiment classification.

### Dataset Statistics

| Split     | Samples |
| :-------- | ------: |
| Training  |  25,000 |
| Test      |  25,000 |
| Unlabeled |  50,000 |

The labeled training data is divided into:

* **22,500** training samples
* **2,500** validation samples
* **25,000** test samples

The test set is kept separate and used for final model evaluation.

The dataset can be loaded directly using Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
```

### Classification Task

The task is binary sentiment classification:

```text
0 → Negative
1 → Positive
```

---

## 🛠️ Technology Stack

| Technology                    | Purpose                       |
| :---------------------------- | :---------------------------- |
| **Python**                    | Core programming language     |
| **PyTorch**                   | Custom LSTM and deep learning |
| **FastAI**                    | ULMFiT / AWD-LSTM             |
| **Hugging Face Transformers** | BERT                          |
| **Hugging Face Datasets**     | IMDb dataset                  |
| **scikit-learn**              | Evaluation metrics            |
| **Pandas**                    | Data manipulation             |
| **Matplotlib**                | Visualization                 |

---

## 📌 Key Techniques Used

* Natural Language Processing
* Text preprocessing
* Tokenization
* Vocabulary construction
* Word embeddings
* Recurrent Neural Networks
* LSTM
* AWD-LSTM
* Transfer learning
* Language-model pretraining
* Progressive unfreezing
* Discriminative learning rates
* Transformer architecture
* BERT fine-tuning
* Gradient clipping
* Early stopping
* Learning-rate scheduling
* Model evaluation

---

## 📈 Training Performance

| Model           | Best Validation Accuracy | Best Epoch |
| :-------------- | :----------------------: | :--------: |
| **Custom LSTM** |        **86.44%**        |      4     |
| **ULMFiT**      |        **91.96%**        |    Final   |
| **BERT**        |        **93.42%**        |    Final   |

The validation results are used during model development, while the test set is reserved for final performance evaluation.

---

## 🔬 Evaluation Metrics

The models are evaluated using multiple classification metrics:

### Accuracy

Measures the overall proportion of correctly classified reviews.

### Precision

Measures how many reviews predicted as positive are actually positive.

### Recall

Measures how many actual positive reviews are correctly identified.

### F1 Score

Provides a balance between precision and recall.

Using multiple metrics provides a more complete view of model performance than accuracy alone.

---

## 🔍 Key Learnings

### 1. Understanding LSTM Internals

Implementing the LSTM manually provided a deeper understanding of:

* Forget gates
* Input gates
* Candidate states
* Output gates
* Hidden states
* Cell states
* Sequence processing

### 2. Importance of Transfer Learning

ULMFiT demonstrated how pretrained language representations can significantly improve downstream NLP performance compared with training an LSTM from scratch.

### 3. Progressive Fine-Tuning

ULMFiT showed the importance of gradually adapting pretrained representations instead of immediately updating the entire network with a large learning rate.

### 4. Contextual Representation

BERT's bidirectional Transformer architecture provides richer contextual representations compared with traditional unidirectional recurrent architectures.

### 5. Model Comparison

The project demonstrates that model architecture and pretraining strategy can have a significant impact on NLP classification performance.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks

Run the notebooks in the following order:

```text
01_Custom_LSTM.ipynb
        ↓
02_ULMFiT_AWD_LSTM.ipynb
        ↓
03_BERT.ipynb
```

The IMDb dataset is downloaded automatically through Hugging Face.

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
fastai>=2.7.0
scikit-learn>=1.2.0
pandas>=1.5.0
matplotlib>=3.6.0
```

---

## 🎯 Project Objective

The primary objective of this project is not only to achieve high classification accuracy, but also to understand how different NLP architectures approach the same sentiment classification problem.

The project progresses through three levels:

```text
LSTM from Scratch
        ↓
ULMFiT Transfer Learning
        ↓
BERT Transformer Fine-Tuning
```

This provides a practical comparison between manually trained recurrent models, pretrained recurrent language models, and modern Transformer-based NLP.

---

## 🏆 Final Comparison

| Aspect                    | Custom LSTM |   ULMFiT   |    BERT    |
| :------------------------ | :---------: | :--------: | :--------: |
| Trained from scratch      |      ✅      |      ❌     |      ❌     |
| Pretrained language model |      ❌      |      ✅     |      ✅     |
| Transfer learning         |      ❌      |      ✅     |      ✅     |
| Recurrent architecture    |      ✅      |      ✅     |      ❌     |
| Transformer architecture  |      ❌      |      ❌     |      ✅     |
| Progressive unfreezing    |      ❌      |      ✅     |      ✅     |
| Test Accuracy             |  **84.28%** | **91.84%** | **93.42%** |
| F1 Score                  |  **0.8440** | **0.9157** | **0.9345** |

---

## 📬 Contact

Feel free to connect with me regarding this project, NLP, Deep Learning, or Data Science.

* **GitHub:** [https://github.com/saravanansokkalingam1997-gif]
* **LinkedIn:** [Your Name](https://linkedin.com/in/your-profile)

---

⭐ **If you found this project useful, consider giving the repository a star!**