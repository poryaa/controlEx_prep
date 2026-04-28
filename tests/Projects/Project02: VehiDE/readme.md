# Automated Vehicle Damage Triage for Insurance Claims

This project explores an end-to-end computer vision pipeline for automated vehicle damage analysis using the VehiDE dataset. It focuses on image-based damage understanding for insurance-style triage, with experiments covering dataset inspection, label conversion, segmentation training with YOLO, and instance segmentation with Mask R-CNN.[1]

## Project overview

The notebook frames the problem as **automated vehicle damage triage**, with a primary focus on identifying and localizing damage categories from vehicle images. The stated design includes a first stage for broad damage presence/type classification and a second stage for damage localization using segmentation-style outputs, which aligns well with insurance claim review workflows.[1]

Rather than treating the task as simple image classification, the project emphasizes structured visual understanding. This is reflected in the use of polygon annotations, conversion pipelines, and segmentation models that can localize damaged regions instead of only predicting image-level labels.[1]

## Data

The project uses the **VehiDE Dataset**, downloaded through `kagglehub`, with separate directories for training images and validation images. The notebook reports **11,621 annotated training images** and **2,324 validation images**, indicating a reasonably sized dataset for damage segmentation experiments.[1]

The raw annotations contain **7 unique labels** in the original dataset. These labels are translated into English through a mapping layer used during conversion, resulting in the following target categories: `paintscratch`, `dented`, `crack`, `missingparts`, `crushedpanel`, `punctured`, and `brokenglass`.[1]

### Dataset inspection

The notebook includes several exploratory checks before training. It reports image widths ranging from **204 to 2365 pixels** with a mean of **1375**, heights ranging from **153 to 2560 pixels** with a mean of **1024**, and color modes dominated by `RGB` with a smaller number of `RGBA` images.[1]

Label counts also show notable class imbalance, with `trayson` / `paintscratch` appearing most often and `vokinh` / `brokenglass` among the least frequent classes. This matters because segmentation and instance segmentation models may favor more frequent damage types unless balancing strategies are introduced later.[1]

## Methodology

The project follows a practical experimental workflow: inspect the dataset, normalize the label space, convert the raw annotation format into model-specific training formats, train segmentation models, and compare outputs qualitatively and quantitatively. Two main modeling tracks appear in the notebook: **YOLO-based segmentation** and **Mask R-CNN-based instance segmentation**.[1]

This methodology is strong for a portfolio-style applied vision project because it shows not only modeling but also data engineering and annotation adaptation. The notebook demonstrates how to move from an unstructured dataset format to reusable training pipelines for multiple model families.[1]

### Data conversion pipelines

A substantial part of the notebook is dedicated to converting the original VehiDE annotations into formats required by downstream frameworks. One pipeline converts the dataset into **YOLO segmentation format** with normalized polygon coordinates and a generated `dataset.yaml`, while another converts it into a **Mask R-CNN dataset structure** with per-image JSON annotation files and metadata.[1]

For Mask R-CNN, polygon annotations are rasterized into binary masks and transformed into bounding boxes using `masks_to_boxes`. This is an important implementation detail because it enables a single annotation source to support both mask prediction and box-based training targets.[1]

## Models implemented

The notebook includes at least two segmentation-oriented model families:

- **YOLO segmentation baseline** using `yolo11n-seg.pt` or `yolov8n-seg.pt`, trained with image size 640, batch size 32, patience 5, and up to 100 epochs.[1]
- **Mask R-CNN** built from `torchvision.models.detection.maskrcnn_resnet50_fpn`, adapted to the VehiDE class set with custom box and mask predictors.[1]

The YOLO branch is aimed at efficient segmentation with straightforward training and validation through Ultralytics. The Mask R-CNN branch is more customizable and highlights deeper understanding of PyTorch datasets, transforms, collate functions, and instance segmentation internals.[1]

## Training setup

The YOLO segmentation training configuration uses cached data loading, GPU training on `device=0`, and early stopping via `patience=5`. This makes the experiment practical for iterative prototyping and avoids overtraining when validation improvement stalls.[1]

The Mask R-CNN training setup uses SGD with learning rate `0.005`, momentum `0.9`, weight decay `0.0005`, a step scheduler, checkpointing of `last.pth` and `best.pth`, and manual early stopping based on epoch loss. This branch is more engineering-heavy and demonstrates a custom training loop rather than only relying on a high-level framework.[1]

## Results

The notebook contains a dedicated **Results** section for the YOLO segmentation model, including validation execution and visualization of a **confusion matrix** from the segmentation run. It also includes qualitative prediction visualization on a test image, where the trained model overlays predicted masks, bounding boxes, class labels, and confidence scores.[1]

Although the searchable notebook summary does not expose the full numeric validation metrics inline, it clearly shows that the project reaches the stage of trained-model validation and inference visualization for YOLO segmentation. The Mask R-CNN section also includes saved checkpoints and epoch-level loss tracking, indicating that training was executed with model selection logic based on best observed loss.[1]

### What the results show

The strongest visible outcome from the notebook is that the project successfully turns raw vehicle-damage polygon annotations into operational segmentation pipelines. That is valuable because the most difficult part of many applied vision projects is often data preparation and label compatibility rather than only model definition.[1]

The notebook also demonstrates that the same dataset can support multiple segmentation paradigms. YOLO offers a faster deployment-oriented baseline, while Mask R-CNN provides a richer instance segmentation workflow with explicit masks and region proposals.[1]

## Repository structure

The current project is primarily organized around a single notebook:

- `main.ipynb` — contains dataset download, exploratory data analysis, label translation, conversion to YOLO and Mask R-CNN formats, training code, validation, and visualization.[1]

During execution, the notebook also creates derived directories such as:

- `yolodataset/` for YOLO-compatible images, labels, and `dataset.yaml`.[1]
- `maskrcnndataset/` for images, converted annotations, and metadata used by the PyTorch dataset class.[1]
- `runs/` directories for training outputs, best weights, validation plots, and checkpoints.[1]

## How to run

1. Create a Python environment with `torch`, `torchvision`, `ultralytics`, `kagglehub`, `Pillow`, `numpy`, `matplotlib`, and related notebook dependencies installed.
2. Open `main.ipynb` in Jupyter Notebook or VS Code.
3. Run the notebook cells in order to download the VehiDE dataset, inspect the data, and generate model-specific dataset formats.
4. Train the YOLO segmentation model or the Mask R-CNN model, depending on the experiment you want to reproduce.
5. Run validation and inference cells to inspect confusion matrices, saved weights, and predicted segmentation overlays.[1]

## Why this project matters

This project is a good example of business-relevant applied AI because it maps directly to insurance claims triage and damage assessment workflows. Instead of limiting the problem to generic object detection, it focuses on fine-grained damage categories and region localization, which are closer to real operational needs in claim review systems.[1]

It also shows production-adjacent thinking: label standardization, annotation conversion, support for multiple training frameworks, and qualitative plus quantitative evaluation. That makes it stronger than a notebook that only trains a model on preprocessed data without addressing real dataset handling challenges.[1]

## Suggested improvements

- Add a concise experiment table with final YOLO validation metrics such as mask mAP, box mAP, precision, and recall.
- Add Mask R-CNN validation metrics on the same validation split for direct model comparison.
- Introduce class-balancing strategies or augmentation targeted at underrepresented damage classes.
- Export a lightweight inference script for single-image insurance-claim triage demos.
- Add error analysis on confusing categories such as dents, crushed panels, and missing parts where visual boundaries may overlap.[1]

## Conclusion

This project demonstrates a full computer vision workflow for vehicle damage analysis, starting from raw annotated images and ending with trained segmentation models and inference outputs. Its main strength is not only the modeling itself, but the complete pipeline design that makes the VehiDE dataset usable for both YOLO segmentation and Mask R-CNN instance segmentation experiments.[1]