# Cattle Classification System Using Deep Learning

Research implementation for individual cattle identification from visual patch patterns using a ResNet-50 image-classification model.

This repository accompanies the publication:

> Sharma, Yuvraj, Shaunik, Shivam, Luthra, Meghna, & Garg, Aditya. (2026). *Cattle Classification System Using Deep Learning*. https://doi.org/10.1201/9781042004607-45

**This is a research reproduction repository, not a packaged production application.** The included Python file is an exported version of the supplied Colab notebook. It preserves the research workflow and should be reviewed and adapted before being run locally.

## Repository contents

```text
cattle-classification-deep-learning/
├── research/
│   └── cattle_identification_resnet50.py
├── docs/
│   └── publication.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
└── README.md
```

## Method summary

- **Task:** multi-class individual cattle identification using visual coat/patch patterns
- **Dataset:** Open Cow 2020 Identification Dataset
- **Training images:** 4,256
- **Test images:** 496
- **Identities/classes:** 48
- **Input size:** 224 × 224 pixels
- **Batch size:** 32
- **Architecture:** ResNet-50, initialized without pretrained weights; final fully connected layer replaced for 48 classes
- **Optimizer:** Adam
- **Learning-rate scheduler:** StepLR (`step_size=5`, `gamma=0.1`)
- **Training duration:** 15 epochs

The notebook reports that no separate validation split was created.

## Reported results

The following values are transcribed from the supplied notebook's README outline. They have **not** been independently reproduced as part of preparing this repository.

| Metric | Training (final epoch) | Test |
|---|---:|---:|
| Accuracy | 0.9946 | 0.9819 |
| Precision | 0.9944 | 0.9701 |
| Recall | 0.9917 | 0.9802 |
| F1-score | 0.9929 | 0.9727 |
| Loss | 0.0274 | Not reported |
| Training time | Approximately 14 min 23 sec | — |

Metric averaging strategy is not specified in the supplied outline. Treat the figures as reported results, not independently verified benchmark claims.

## Dataset

The source notebook references the **Open Cow 2020 Identification Dataset** and expects an ImageFolder-style directory structure:

```text
dataset/
└── identification/
    └── images/
        ├── train/
        │   ├── identity_1/
        │   └── ...
        └── test/
            ├── identity_1/
            └── ...
```

The dataset is not included in this repository. Obtain it from its original source and check its terms of use before downloading or redistributing it.

## Environment setup

Python 3.8 or newer is suggested. Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by operating system and CUDA version. If the default installation does not match your hardware, use the official PyTorch installation selector.

## Running the research code

The exported source currently contains Kaggle-specific absolute dataset paths. Update `train_data_path` and `test_data_path` in `research/cattle_identification_resnet50.py` to point to your local dataset folders before execution.

```bash
python research/cattle_identification_resnet50.py
```

The file is an export of a notebook and may require adaptation for a local run. In particular:

1. The optimizer is referenced as `optim.Adam`, so ensure `import torch.optim as optim` is present.
2. The inference loop treats the second `ImageFolder` batch value as filenames. `ImageFolder` returns class labels there, not filenames; adjust the loop if filenames are required.
3. The notebook's reported results are included for documentation, but running the exported file is not guaranteed to reproduce them without resolving the environment and code issues above.
4. A separate validation set was not used in the supplied workflow.

## Model weights and generated outputs

The notebook saves weights as `resnet_cow.pth`. Model weights, dataset images, and generated plots are not included here. Add them only if you have the right to redistribute them, and document how they were produced.

## Publication and authorship

This work is associated with the publication listed above. The publication has four authors. This repository credits all listed authors and does not claim sole authorship.

See [docs/publication.md](docs/publication.md) and [CITATION.cff](CITATION.cff) for citation details.

## Limitations and future work

- Add a dedicated validation split and document the split protocol.
- Record the precision/recall/F1 averaging method.
- Add reproducible environment and random-seed details.
- Validate the exported training and inference code end to end.
- Compare against pretrained backbones and report inference latency.

## Acknowledgements

The research uses the Open Cow 2020 Identification Dataset. Please cite and follow the dataset provider's terms where applicable.
