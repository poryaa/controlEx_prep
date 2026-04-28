# Car Damage Classification for Insurance and Vehicle Inspection

This project builds and compares multiple deep learning models for multiclass car-damage image classification, with a focus on practical deployment for insurance claim automation, vehicle inspection, and fleet management. The notebook evaluates transfer-learning CNN baselines and a YOLO classification model on the Kaggle Car Damage Assessment dataset.[1]

## Project overview

The core task is **8-class vehicle damage classification** from images. Rather than only training one model, the project is structured as a comparative study across several architectures and training strategies, which makes it useful both as a research exercise and as a portfolio-ready applied computer vision project.[1]

The notebook explicitly highlights a business use case: automated damage understanding for operational workflows such as claims triage and inspection support. That gives the project a clear real-world framing beyond pure academic benchmarking.[1]

## Dataset

The project uses the **Kaggle Car Damage Assessment** dataset with **1,543 images** in total. The notebook states an **80/15/5 split**, resulting in **1,225 training images**, **238 validation images**, and **80 test images**.[1]

The classification problem contains **8 classes**: `Broken window`, `Broken headlight`, `Damage`, `Door scratch`, `Bumper dent`, `Door dent`, `Glass shatter`, and `Unknown`.[1]

### Data preparation

The dataset is downloaded through `kagglehub`, explored with label-distribution plots and sample images, and then reorganized into a folder structure compatible with both Keras directory datasets and YOLO classification training. This is important because it shows the project handles the full data pipeline rather than assuming a preformatted training split.[1]

The notebook also includes checks of image dtype, shape, and pixel range, along with class distribution visualizations across train, validation, and test splits. Those steps help verify that the dataset structure and label coverage are suitable before training begins.[1]

## Methodology

The project uses **transfer learning** as its main strategy, comparing multiple pretrained backbones with different fine-tuning depths and augmentation policies. The workflow moves from a simple frozen baseline to progressively stronger fine-tuned models, then adds a YOLO classification model as a production-oriented benchmark.[1]

This is a strong methodology because it tests not only architecture choice but also the impact of preprocessing, augmentation, and trainable-layer selection. The notebook turns model development into a structured experiment rather than a single training run.[1]

## Models compared

The notebook evaluates five model variants:

| Model | Architecture | Strategy | Test performance |
|---|---|---|---|
| M1 | EfficientNetB0 | Frozen baseline | 36.25% accuracy, 1.8034 loss [1] |
| M2 | EfficientNetB0 | Augmentation + fine-tune last 10 layers | 82.50% accuracy, 0.5425 loss [1] |
| M3 | MobileNetV3-Large | Augmentation + fine-tune last 20 layers | 78.75% accuracy, 0.6239 loss [1] |
| M4 | ResNet50 | Augmentation + fine-tune last 20 layers | 85.00% accuracy, 0.6065 loss [1] |
| M5 | YOLO11s-cls | Full fine-tuning with built-in preprocessing and augmentation | 87.50% accuracy [1] |

The Keras models are built with pretrained ImageNet weights and trained using directory-based datasets created via `tf.keras.utils.image_dataset_from_directory`. The YOLO branch is trained with Ultralytics classification mode, which simplifies preprocessing and experiment management.[1]

## Training setup

For the Keras models, the notebook uses `224x224` inputs, batch size `32`, Adam optimization, sparse categorical cross-entropy loss, early stopping on validation loss, and image augmentation including horizontal flips, rotation, zoom, height changes, and width changes. Dropout is also used in fine-tuned models for regularization.[1]

The YOLO11s classification experiment is trained with image size `224`, batch size `32`, Adam optimizer, early stopping with `patience=5`, and augmentation settings including horizontal flip, rotation, translation, scale, and HSV variation. The notebook reports that this model stopped after **24 epochs** and achieved the best final performance.[1]

## Key finding: preprocessing matters

One of the most important findings in the notebook is that the original frozen EfficientNet baseline failed badly because of a **preprocessing mismatch**. The notebook explains that ImageNet-pretrained weights expect normalized inputs, while the broken baseline effectively received incorrectly scaled data, causing feature mismatch and poor convergence.[1]

This is why M1 reached only **36.25% accuracy**, far below the fine-tuned models and even only modestly above the random baseline for 8 classes. The notebook presents this as a critical lesson: transfer learning is highly sensitive to correct input preprocessing.[1]

### Why other models performed better

The notebook notes that the fine-tuned Keras models still recovered useful performance because unfreezing layers allowed them to adapt to the imperfect input pipeline. That explains why M2 to M4 still reached roughly **79% to 85%** accuracy despite the preprocessing issue in the earlier setup.[1]

The YOLO model performed best largely because its framework handles preprocessing automatically and couples that with strong augmentation and full fine-tuning. This reduces the chance of hidden configuration errors and makes the training setup more robust for deployment-style use.[1]

## Results

The final ranking reported in the notebook is:

1. **M5 - YOLO11s-cls**: 87.50% accuracy.[1]
2. **M4 - ResNet50 Aug FT**: 85.00% accuracy, 0.6065 loss.[1]
3. **M2 - EfficientNetB0 Aug FT**: 82.50% accuracy, 0.5425 loss.[1]
4. **M3 - MobileNetV3-Large Aug FT**: 78.75% accuracy, 0.6239 loss.[1]
5. **M1 - EfficientNetB0 Baseline**: 36.25% accuracy, 1.8034 loss.[1]

The notebook also saves a CSV file with model comparison results and generates a bar chart of test accuracy. In addition, it computes per-class accuracies, which gives a more granular view than overall accuracy alone.[1]

### Per-class observations

The per-class outputs show that some categories are easier than others across models. For the best YOLO model, `bumperdent` and `bumperscratch` reach **100%**, while harder categories such as `doordent` and `headlamp` are lower, at **66.67%** and **71.43%** respectively.[1]

This class-level breakdown is useful because overall accuracy can hide uneven performance. In a real insurance workflow, weaker categories may require targeted augmentation, more data, or relabeling improvements before deployment.[1]

## Evaluation workflow

The notebook includes confusion matrices for the Keras models and a custom evaluation routine for YOLO classification on the test set. It also unifies the evaluation output into a common comparison summary, which is a good design choice when benchmarking heterogeneous frameworks.[1]

Results are logged to MLflow as well, although the notebook shows permission warnings during autologging. Even so, the use of MLflow indicates an experiment-tracking mindset and makes the project more aligned with production ML practices.[1]

## Repository contents

The main project logic is contained in:

- `main.ipynb` — dataset acquisition, exploration, train/validation/test split creation, model training, evaluation, confusion matrices, MLflow setup, and comparison plots.[1]

The notebook also creates artifacts such as:

- `dataset/` with train, validation, and test folder structure for classification.[1]
- `model1.keras` to `model4.keras` for saved Keras models.[1]
- `runs/classify/` for YOLO training outputs and best weights.[1]
- `modelcomparisonresults.csv` and `modelcomparison.png` for summary reporting.[1]

## How to run

1. Install Python dependencies including `tensorflow`, `ultralytics`, `mlflow`, `kagglehub`, `pandas`, `numpy`, `matplotlib`, and `Pillow`.[1]
2. Open `main.ipynb` in Jupyter or VS Code.[1]
3. Run the data download and split-generation cells first to prepare the `dataset/` directory.[1]
4. Train the Keras models M1 to M4 and the YOLO model M5.[1]
5. Run the evaluation cells to generate test metrics, confusion matrices, CSV summaries, and the model-comparison visualization.[1]

## Why this project is strong

This project demonstrates several important engineering skills at once: data handling, transfer learning, augmentation design, model comparison, test-set evaluation, MLflow usage, and practical error analysis. Its biggest strength is that it does not only report a best score, but also explains **why** one model failed and another succeeded.[1]

That makes the notebook especially compelling for a portfolio, because it shows debugging ability and experimental reasoning. Identifying the preprocessing mismatch in the baseline is as valuable as the final model score itself.[1]

## Suggested improvements

- Retrain all Keras models with consistent preprocessing layers included directly in each architecture.
- Add precision, recall, and F1-score per class, not only accuracy and confusion matrices.
- Introduce class-balanced sampling or weighted loss if some categories are underrepresented.
- Add Grad-CAM or saliency visualizations to make predictions easier to interpret.
- Export the best YOLO classification model behind a simple inference API or demo app.[1]

## Conclusion

The project shows that well-configured transfer learning can solve multiclass car-damage classification effectively, with YOLO11s-cls achieving the best reported test accuracy at **87.50%**. Just as importantly, it shows that preprocessing and training design strongly affect transfer-learning outcomes, making experiment setup a central part of model success.[1]