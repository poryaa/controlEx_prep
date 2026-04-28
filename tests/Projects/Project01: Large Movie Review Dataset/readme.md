# Binary Text Classification on the IMDB Large Movie Review Dataset

This project builds and compares several deep learning approaches for binary sentiment classification on movie reviews from the IMDB Large Movie Review Dataset. The objective is to predict whether a review expresses a positive or negative sentiment using TensorFlow/Keras-based text models trained on raw review text.[1]

## Project overview

The notebook focuses on a supervised binary classification task using the `aclImdb` dataset, which is a standard benchmark for sentiment analysis. The workflow includes dataset download and extraction, inspection of raw text samples, train/validation/test preparation, model training, and performance comparison across multiple architectures.[1]

## Data

The project uses the IMDB Large Movie Review Dataset stored under the `aclImdb` directory structure with separate `train` and `test` folders and class labels `pos` and `neg`. The notebook counts 12,500 positive and 12,500 negative examples in the training split, and the same class balance in the original test split, for a total of 50,000 labeled reviews.[1]

For modeling, all 25,000 training reviews are loaded into memory as the training set. The original 25,000-test-review split is further divided into a validation set of 15,000 reviews and a final test set of 10,000 reviews using `train_test_split(..., random_state=42, stratify=labels_test)`, which preserves class balance during the split.[1]

### Data preparation

The notebook performs minimal manual preprocessing because the text pipeline is handled inside the models. Raw reviews are passed directly into TensorFlow datasets, and tokenization is performed with a `TextVectorization` layer configured with `max_tokens=20000` and `output_sequence_length=300` for the token-based models.[1]

The data pipeline uses `tf.data.Dataset.from_tensor_slices`, shuffling for the training set, batching with `BATCH_SIZE=64`, and `prefetch(tf.data.AUTOTUNE)` to improve throughput during training. The Universal Sentence Encoder experiment also adds dataset caching before batching and prefetching.[1]

## Methodology

The project follows an experimental comparison design: several neural architectures are trained on the same task and then compared by validation and test accuracy. All models use binary cross-entropy loss, the Adam optimizer, and accuracy as the main evaluation metric, while early stopping monitors validation loss and restores the best weights.[1]

TensorBoard callbacks are included for training logs, and each experiment is trained for up to 100 epochs with patience-based early stopping. This setup makes the comparison more reliable because each model is evaluated under a similar training procedure.[1]

### Models implemented

The notebook evaluates five model families:

- **Model 1: Baseline embedding model** — `TextVectorization` → `Embedding(20000, 128)` → `GlobalAveragePooling1D` → `Dense(64, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)`.[1]
- **Model 2: Regularized smaller embedding model** — `TextVectorization` → `Embedding(20000, 64)` → `GlobalAveragePooling1D` → `Dense(32, relu, L2)` → `Dropout(0.5)` → `Dense(1, sigmoid)`.[1]
- **Model 3: BiLSTM model** — `TextVectorization` → `Embedding(20000, 128)` → `Bidirectional(LSTM(64, return_sequences=True))` → `GlobalMaxPooling1D` → dense layers with dropout.[1]
- **Model 4: CNN model** — `TextVectorization` → `Embedding(20000, 128)` → parallel `Conv1D` branches with kernel sizes 3, 4, and 5, followed by global max pooling, concatenation, dropout, and dense layers.[1]
- **Model 5: Universal Sentence Encoder model** — raw text → TensorFlow Hub Universal Sentence Encoder → dropout and dense classifier head.[1]

## Results

The strongest test result in the notebook comes from the CNN-based architecture, which reaches a test accuracy of **0.8839**. The BiLSTM model follows closely with **0.8757**, the smaller regularized embedding model reaches **0.8769**, the baseline embedding model reaches **0.8695**, and the Universal Sentence Encoder model achieves **0.8657**.[1]

### Model performance

| Model | Main idea | Test accuracy |
|---|---|---|
| Baseline embedding model | Average pooled learned token embeddings with dense classifier | 0.8695 [1] |
| Smaller regularized embedding model | Lower-capacity embedding model with L2 regularization and higher dropout | 0.8769 [1] |
| BiLSTM model | Sequence model with bidirectional recurrent context modeling | 0.8757 [1] |
| CNN model | Parallel convolution filters over token sequences | 0.8839 [1] |
| USE model | Pretrained sentence embedding from TensorFlow Hub with classifier head | 0.8657 [1] |

The results suggest that the convolutional architecture captured local sentiment patterns particularly well on this dataset, outperforming both the simpler embedding baselines and the pretrained sentence embedding approach in this notebook. The BiLSTM and CNN models also show a common pattern: very high training accuracy but weaker validation improvements after the best epoch, which indicates some degree of overfitting controlled partly by early stopping.[1]

## Interpretation

A useful takeaway from this project is that more complex models do not automatically guarantee better generalization. Although recurrent and convolutional models have higher representational power, the best result comes from a relatively compact text CNN rather than the largest or most computationally expensive model.[1]

Another practical observation is that the smaller regularized embedding model performs competitively despite having fewer parameters than the baseline embedding model. That makes it an attractive option when training speed, memory footprint, or deployment simplicity matters more than squeezing out the absolute best accuracy.[1]

## Repository structure

The current project is centered around the notebook below:

- `01_bin_classification.ipynb` — main experiment notebook containing data loading, preprocessing, model definitions, training, and evaluation.[1]

## How to run

1. Create and activate a Python environment with TensorFlow, scikit-learn, and TensorFlow Hub installed.
2. Open `01_bin_classification.ipynb` in Jupyter Notebook or VS Code.
3. Run the cells in order to download/extract the IMDB dataset, prepare the splits, and train the models.
4. Optionally launch TensorBoard on the generated log directory to inspect learning curves.[1]

## Suggested improvements

- Add richer evaluation metrics such as precision, recall, F1-score, ROC-AUC, and confusion matrices for a more complete binary classification analysis.
- Save trained model artifacts and preprocessing configuration for reproducibility and inference.
- Introduce hyperparameter search over vocabulary size, sequence length, embedding dimension, dropout, and learning rate.
- Compare the deep learning models with classical baselines such as logistic regression or linear SVM on TF-IDF features.
- Add error analysis on misclassified reviews to better understand failure cases.[1]

## Conclusion

This project demonstrates a complete sentiment analysis workflow on the IMDB dataset using TensorFlow/Keras, from raw text ingestion to multi-model comparison. Among the tested approaches, the CNN model delivers the best test accuracy in the notebook, while the smaller regularized embedding model offers a strong balance between simplicity and performance.[1]