# Sign-Language-Recognition-using-Temporal-Classification

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Boffin-Omega/Sign-Language-Recognition-using-Temporal-Classification.git
   cd Sign-Language-Recognition-using-Temporal-Classification
   ```

2. Upload the `data` folder to your Google Drive.

## Phase 1: Preprocessing

Open `preprocessing.ipynb` in Google Colab.

This notebook loads and cleans the raw sign-language dataset, performs temporal and spatial normalization, creates the session-based train/test split, and generates the processed datasets required for model training.

Run all code cells. If Colab asks for permission to access Google Drive, grant the requested permission.

## Phase 2: Model Training and Evaluation

### SVM

Open `svm.ipynb` in Google Colab.

This notebook loads the preprocessed classical ML data, trains Linear and RBF SVM models, and evaluates them using accuracy, precision, recall, F1-score, and confusion matrices. Training and test accuracy are also recorded.

Run all code cells. If Colab asks for permission to access Google Drive, grant the requested permission.

### LSTM

Open `lstm.ipynb` in Google Colab.

This notebook loads the preprocessed sequential data, trains LSTM models with different numbers of recurrent layers, and evaluates them using accuracy, precision, recall, F1-score, and confusion matrices. Training and validation accuracy and loss are also recorded.

Run all code cells. If Colab asks for permission to access Google Drive, grant the requested permission.

## Methodology

The raw recordings are variable-length time series containing multiple features per time step. Temporal normalization converts the recordings to a fixed sequence length, followed by spatial normalization to a common scale. The dataset is split by recording sessions to separate training and testing data.

For the SVM models, the normalized sequences are flattened into fixed-length feature vectors before classification using Linear and RBF SVMs.

For the LSTM models, the temporal structure is preserved and the sequences are directly used as input to LSTM architectures with different recurrent depths. The models are compared using the same evaluation metrics.
