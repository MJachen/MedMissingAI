# MedMissingAI Project Plan

## 1. Project Goal

Build a research and portfolio-oriented prototype system for incomplete multimodal medical image classification and explainability.

The system should support incomplete multimodal medical image inputs, such as missing MRI modalities, and output disease classification probabilities together with visualization maps showing image regions that contribute strongly to the model prediction.

This project is intended for research training, job portfolio demonstration, and learning the full medical imaging AI workflow. It is not intended for clinical diagnosis or direct clinical decision-making.

## 2. Target Workflow

```text
Incomplete multimodal medical images
-> input validation
-> preprocessing
-> modality missingness detection
-> tensor construction
-> classification model
-> prediction probabilities
-> explainability heatmap
-> visual report / software demo
```

## 3. Overall Timeline

| Stage | Time | Goal | Main Tasks | Deliverables |
|---|---:|---|---|---|
| 0. Project initialization | Week 1 | Define project scope and create repository | Define disease task, input modalities, missing modality setting, data format, repository structure, initial README | Runnable empty project framework |
| 1. Data pipeline | Weeks 2-4 | Convert medical image files into model-ready tensors | Support `.nii.gz` loading, build `manifest.csv`, implement resampling, cropping, normalization, modality mask generation, shape checking | `preprocess.py`, `dataset.py`, example manifest |
| 2. Missing-modality classification baseline | Weeks 5-8 | Implement input-incomplete-modality to classification-probability baseline | Build multimodal Dataset, modality dropout, simple multi-encoder or shared-encoder classifier, train/validation/test scripts | Stable baseline and initial result table |
| 3. Inference pipeline | Weeks 9-10 | Turn training code into callable single-case inference | Define `predict_case()`, load checkpoint, output class probabilities and missing modality information | Single-case inference script |
| 4. Explainability module | Weeks 11-14 | Generate heatmaps for regions important to classification | Implement 3D Grad-CAM, optionally add occlusion sensitivity, generate slice heatmaps and overlay images, run sanity checks | Heatmaps, overlays, explainability demo |
| 5. Modality completion module | Weeks 15-18 | Add the research-specific missing modality completion component | Start with feature-level completion, compare no completion / feature completion / image completion, evaluate whether completion improves classification | Completion model and ablation results |
| 6. Software prototype | Weeks 19-22 | Build an interactive local demo | Use Gradio or Streamlit for image upload, missing modality display, prediction probability display, heatmap visualization, and report download | Local runnable software demo |
| 7. Validation and documentation | Weeks 23-26 | Make the project portfolio-ready | Test different missing modality combinations, organize README, technical document, demo video, limitations, and result tables | Complete project repository and technical documentation |

## 4. Current Immediate Stage: Stage 0

Because the project has just started, the current focus is project initialization rather than model development.

### Week 1 Tasks

1. Decide the project name, for example `MedMissingAI` or `missing-modal-medai`.
2. Decide the first target task: binary classification or multiclass classification.
3. Decide the first set of input modalities, for example `T1`, `T1ce`, `T2`, and `FLAIR`.
4. Decide which missing modality combinations should be supported in the first version.
5. Decide whether to use a public dataset first or a private dataset after de-identification.
6. Create the initial repository structure.
7. Create `README.md` with project goal, task definition, and non-clinical-use statement.
8. Create `data/example_manifest.csv` to define the expected data format.
9. Create a simple image loading and shape-checking script.

### Week 1 Success Criteria

By the end of Week 1, the following questions should be answerable:

```text
What is the model input?
What is the model output?
How are missing modalities represented?
Where does each case read image paths and labels from?
What tensor shape will the model receive?
```

## 5. Suggested Repository Structure

```text
MedMissingAI/
├── README.md
├── project_plan.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── preprocess.yaml
│   ├── train_baseline.yaml
│   └── infer.yaml
├── data/
│   ├── README.md
│   └── example_manifest.csv
├── src/
│   ├── datasets/
│   ├── preprocessing/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   ├── engine/
│   ├── inference/
│   ├── explainability/
│   ├── visualization/
│   └── utils/
├── scripts/
│   ├── preprocess.py
│   ├── check_images.py
│   ├── train.py
│   ├── test.py
│   └── infer_one_case.py
├── experiments/
│   └── README.md
├── docs/
│   └── technical_notes.md
└── tests/
    ├── test_dataset.py
    └── test_model_forward.py
```

## 6. Weekly Review Template

Use this template for the weekly Friday review.

```text
Current stage:
Planned tasks this week:
Completed tasks:
Unfinished tasks:
Main blockers:
Next week's priority tasks:
Does the timeline need adjustment:
```

## 7. Basic Requirements

### Data Requirements

- A legal and de-identified multimodal medical imaging dataset.
- Prefer NIfTI format, such as `.nii.gz`, for the first version.
- Each case should have a subject ID, modality paths, classification label, and split information.
- Private clinical data must be de-identified before use.

### Computing Requirements

- A GPU is strongly recommended for 3D medical image models.
- If GPU memory is limited, start with small patches, small models, and small batch sizes.

### Software Stack

Recommended first-version stack:

```text
Python
PyTorch
MONAI
SimpleITK or NiBabel
scikit-learn
matplotlib
Gradio or Streamlit
```

### Knowledge Requirements

This project should be used to learn the following topics step by step:

```text
medical image preprocessing
multimodal data loading
missing modality representation
classification training and evaluation
checkpoint and inference design
3D Grad-CAM and explainability
software prototype construction
medical AI validation and documentation
```

## 8. Guiding Principle

The project should follow this order:

```text
first close the minimal loop
then improve the model
then add explainability
then build the interface
then strengthen validation and documentation
```

Do not implement classification, completion, explainability, and software UI all at once. The first priority is to make `input images -> prediction probability` stable and understandable.
