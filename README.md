# Pathology Fine-Tuning Benchmark (path-ft-bench)

A benchmarking and adaptation pipeline for pathology foundation models on H&E colorectal histology, validated against an independent external patient cohort.

## Purpose

Takes a pretrained pathology foundation model, fine-tunes it for tissue classification, and checks whether it actually generalizes — evaluated not just on held-out data from the same source, but on a completely separate patient cohort it's never seen. Built end-to-end on public data (NCT-CRC-HE-100K / CRC-VAL-HE-7K), at a scale that runs on a single consumer GPU.

## Approach

The project runs a two-stage comparison of pathology foundation models for diagnostic histopathology image classification.

**Status**

*Complete (MVP):* 
- Dataset exploration completed. Basic utilities generated for visualization and data exploration. Project structure setup.

*In progress / planned:*
- **Stage 1 frozen-probe benchmark**: H-Optimus-0 vs. ImageNet-ViT control, raw vs. Macenko-normalized, single seed"]
- **Additional encoders** for the Stage 1 comparison: CTransPath, UNI, or Prov-GigaPath, if access and time allows.
- **Stain-augmented condition**: random stain-color jitter during training, compared against Macenko normalization as an alternative strategy for handling staining variation.
- **Multi-seed + confidence intervals**: repeat each experiment across 3-5 random seeds to confirm differences between models are real, not run-to-run noise.
- **Stain-swap counterfactual**: re-normalize a correctly-classified image to a different lab's stain profile and re-test — a flipped prediction would mean the model is partly relying on stain color rather than tissue structure.
- **Cohort-separability audit**: check whether a simple classifier can tell training-cohort vs. external-cohort patches apart from their embeddings alone — easy separability would signal a batch-effect shortcut rather than genuine biological generalization.

**Stage 1 — Frozen representation benchmark.** A candidate pathology-pretrained encoder (plus an ImageNet-pretrained control, to isolate the value of domain-specific pretraining) produces embeddings for every tissue patch. A linear probe is trained on top of each set of frozen embeddings and evaluated under matched conditions — raw images vs. Macenko-normalized images — to isolate how much normalization actually contributes.

**Stage 2 — Fine-tuned adaptation.** The Stage 1 encoder is adapted further via parameter-efficient fine-tuning (LoRA) and compared directly against its own frozen-probe baseline, reporting the result in whichever direction it falls.

Every model is evaluated both on held-out data from the same cohort it was tuned on and on a separate external patient cohort it has never seen — the more relevant test of whether a result generalizes beyond the data it was built on.

## Data

**[NCT-CRC-HE-100K](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/NCT-CRC-HE-100K.md)** (training) and **[CRC-VAL-HE-7K](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/NCT-CRC-HE-100K.md)** (external validation) — public colorectal histology datasets (Kather et al.), 224×224 H&E patches across nine tissue classes (tumour epithelium, stroma, lymphocytes, normal mucosa, mucus, smooth muscle, debris, adipose, background), at a resolution and magnification matching how these foundation models were pretrained. The two cohorts are constructed from non-overlapping patients per the dataset's original documentation; patch-level patient IDs aren't included in this release, so that separation is a documented guarantee rather than something independently re-verified here. The external cohort is held out entirely until final evaluation.

Download [NCT-CRC-HE-100K](https://zenodo.org/records/1214456/) with:
```bash
pixi run curl -L -C - -o data/NCT-CRC-HE-100K.zip "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1"
pixi run curl -L -C - -o data/NCT-CRC-HE-100K-NONORM.zip "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K-NONORM.zip?download=1"
```

Download [CRC-VAL-HE-7K](https://zenodo.org/records/1214456/) with:
```bash
pixi run curl -L -C - -o data/CRC-VAL-HE-7K.zip "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1"
```

**Models used:**

Current pathology foundation models such as H-Optimus-0, evaluated alongside an ImageNet-pretrained ViT as a baseline control. 

Additional encoders (CTransPath, UNI, Prov-GigaPath) are added as access is confirmed — see Status above.

## Validation and interpretability

Accuracy alone doesn't establish that a model has learned real tissue biology rather than incidental lab-specific artifacts. The pipeline checks this along several axes, at different stages of completeness (see Status above):

- **Where a prediction comes from** — attention and gradient-based saliency maps show which regions of a tissue image drove each classification.
- **Whether performance holds outside the training cohort** — every headline number is reported on the external patient cohort, not just the cohort used for development.
- *(Planned)* **Whether a prediction is stain-driven or morphology-driven** — re-normalizing a correctly-classified image to a different lab's stain profile and re-testing; a flipped prediction would indicate the model is partly relying on stain color rather than tissue structure.
- *(Planned)* **Whether a model relies on lab-specific shortcuts** — checking how easily a simple classifier can tell which cohort an embedding came from; clean separability would suggest a batch-effect shortcut rather than genuine biological generalization.

## Repository structure

```bash
path-ft-bench/
├── README.md
├── pixi.toml / pixi.lock
├── assets/ # committed figures and small sample-patch subsets
├── src/
│ └── utilities/
│ ├── data_io.py # extraction, metadata, class counts, channel stats
│ └── viz.py # plotting and visualization helpers
├── notebooks/ # exploration, normalization, training, results
└── data/ # gitignored — pointer scripts only, bulk data not committed
```
