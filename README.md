# Lightweight Dual-Modal ECG–PCG Abnormality Detection

Binary cardiac abnormality detection by jointly analysing paired:

- **ECG** — electrical heart activity
- **PCG** — heart-sound signal

This repository contains a Jupyter Notebook implementation of a lightweight, dual-input **PACFNet-style** deep-learning model. The proposed system learns complementary representations from synchronized ECG and PCG beat segments, fuses the two learned embeddings, and predicts whether each cardiac beat belongs to the **Normal** or **Abnormal** class.

> **Important:** This repository is an experimental research implementation and is not a clinical diagnostic tool. The reported results are dataset- and split-dependent and should not be used for medical decision-making.

## Repository contents

| File | Description |
|---|---|
| [`ML_PACFNET2.ipynb`](./ML_PACFNET2.ipynb) | End-to-end notebook for dataset preparation, signal processing, data cleaning, model training, and evaluation |

## Proposed model at a glance

The complete pipeline is:

```text
PhysioNet 2016 ECG/PCG records
              │
              ▼
       Load paired signals
              │
              ▼
    Resample to 2,000 Hz
              │
              ▼
      Modality-specific filtering
       ┌──────────────┴──────────────┐
       ▼                             ▼
 ECG: 0.5–20 Hz                PCG: 25–400 Hz
       └──────────────┬──────────────┘
                      ▼
       Per-record z-score normalization
                      │
                      ▼
       ECG R-peak detection with Christov
                      │
                      ▼
     Synchronized 1,600-sample beat windows
                      │
              ┌───────┴───────┐
              ▼               ▼
        ECG Conv1D branch  PCG Conv1D branch
              └───────┬───────┘
                      ▼
           Concatenated multimodal embedding
                      │
                      ▼
       Dense classifier with sigmoid output
                      │
                      ▼
          Normal / Abnormal prediction
```

## 1. Dataset and labels

The notebook uses the **PhysioNet/CinC Challenge 2016** heart-sound dataset through a Kaggle mirror. It searches for the dataset in common Colab and Kaggle locations and downloads it when it is not already available.

The notebook expects a Kaggle API credential when a download is required:

```text
/content/physionet2016/training-a
/content/physionet2016
/kaggle/input/physionet-challenge-2016
```

Labels are inferred from comments in each `.hea` header file:

- `Normal` → class `0`
- `Abnormal` → class `1`

The ECG and PCG channels are read from each WFDB record as:

- channel `0`: PCG
- channel `1`: ECG

## 2. Signal loading and resampling

Each record is loaded with `wfdb.rdrecord`. Because source records may use different sampling frequencies, both modalities are resampled to a common target frequency of **2,000 Hz** using `scipy.signal.resample`.

Using one sampling rate for both modalities ensures that ECG-derived beat boundaries can be applied to the corresponding PCG signal without losing temporal alignment.

## 3. Modality-specific preprocessing

ECG and PCG have different frequency characteristics, so the notebook processes them with separate fourth-order Butterworth band-pass filters.

### ECG filtering

```text
Band-pass range: 0.5–20 Hz
```

This preserves the main cardiac electrical morphology while reducing baseline drift and high-frequency noise.

### PCG filtering

```text
Band-pass range: 25–400 Hz
```

This preserves the principal heart-sound energy while reducing slow baseline variation and frequencies outside the intended acoustic range.

### Normalization

After filtering, each signal is standardized independently:

```text
x_normalized = (x - mean(x)) / (std(x) + 1e-8)
```

The small epsilon prevents division by zero for nearly constant signals.

## 4. Beat synchronization and segmentation

The notebook uses the filtered ECG as the timing reference. R-peaks are detected with the BioSPPy implementation of the **Christov ECG segmentation algorithm**.

For every detected R-peak, the same time window is extracted from both modalities:

| Window component | Duration | Samples at 2,000 Hz |
|---|---:|---:|
| Before R-peak | 0.3 s | 600 |
| After R-peak | 0.5 s | 1,000 |
| Total | 0.8 s | 1,600 |

Therefore, each paired example has the shape:

```text
ECG: (1600, 1)
PCG: (1600, 1)
```

A beat is retained only when the complete window lies inside the record. If R-peak detection fails or no valid beat is produced, the record is skipped.

The generated compressed dataset contains:

```text
X_ecg       ECG beat windows
X_pcg       synchronized PCG beat windows
y            binary labels
record_ids  source record identifier for every beat
```

## 5. Data quality and sanitization

The notebook performs several validity checks before training:

1. Detects NaN values and infinities.
2. Checks that each beat has finite ECG and PCG values.
3. Removes beats with almost-zero variance.
4. Replaces any remaining invalid values with zero as a final safeguard.
5. Clips extreme values to the range `[-10, 10]` during cleaning.
6. Applies an additional training-time clip to `[-8, 8]`.

The recorded notebook run produced the following intermediate dataset statistics:

| Stage | ECG shape | PCG shape | Class counts | Unique records |
|---|---:|---:|---:|---:|
| Raw beat dataset | `(6975, 1600, 1)` | `(6975, 1600, 1)` | `[2699, 4276]` | `186` |
| Clean beat dataset | `(6611, 1600, 1)` | `(6611, 1600, 1)` | `[2413, 4198]` | `175` |

The cleaning step removed **364 invalid beats**.

## 6. Train/test preparation

The notebook uses an 80/20 stratified split:

```text
Training beats: 5,288
Testing beats: 1,323
```

The original class distribution is imbalanced. To address this in the training set, the minority class is randomly oversampled until both classes have the same number of examples:

```text
Balanced training counts: [3358, 3358]
```

The test set is not oversampled and retains its original distribution.

### Paired augmentation

The notebook creates one augmented copy of every balanced training pair. ECG and PCG are transformed together using:

- a shared random circular time shift between `-20` and `+20` samples
- independent Gaussian noise with standard deviation `0.005`
- final clipping to `[-8, 8]`

The same shift is applied to both modalities so their temporal relationship is preserved.

The final augmented training set contains:

```text
13,432 paired examples
6,716 Normal examples
6,716 Abnormal examples
```

## 7. Dual-branch neural architecture

The model is implemented with TensorFlow/Keras and has two independent one-dimensional convolutional branches.

### Inputs

```text
ECG_Input: (1600, 1)
PCG_Input: (1600, 1)
```

The branches do not share weights. This is important because ECG morphology and PCG acoustics have different signal characteristics and may require different feature detectors.

### ECG branch

The ECG branch uses `base=40` filters:

| Layer | Configuration | Purpose |
|---|---|---|
| Conv1D | 40 filters, kernel size 15, same padding, ReLU | Captures broader ECG morphology and local waveform structure |
| Batch normalization | — | Stabilizes intermediate activations |
| MaxPooling1D | pool size 4 | Reduces temporal resolution and computation |
| Conv1D | 80 filters, kernel size 7, same padding, ReLU | Learns increasingly specific mid-level patterns |
| Batch normalization | — | Improves optimization stability |
| Dropout | 0.15 | Regularizes the branch |
| Conv1D | 160 filters, kernel size 3, same padding, ReLU | Extracts fine-grained local features |
| Batch normalization | — | Stabilizes the final convolutional representation |
| Global average pooling | — | Summarizes distributed activation patterns |
| Global max pooling | — | Preserves the strongest detected response |
| Concatenation | average + max features | Produces the ECG embedding |

### PCG branch

The PCG branch uses `base=56` filters, giving it a slightly wider capacity for heart-sound structure:

| Layer | Configuration | Purpose |
|---|---|---|
| Conv1D | 56 filters, kernel size 15, same padding, ReLU | Captures broad acoustic patterns |
| Batch normalization | — | Stabilizes activations |
| MaxPooling1D | pool size 4 | Reduces temporal resolution |
| Conv1D | 112 filters, kernel size 7, same padding, ReLU | Learns mid-level sound patterns |
| Batch normalization | — | Improves training stability |
| Dropout | 0.15 | Reduces overfitting |
| Conv1D | 224 filters, kernel size 3, same padding, ReLU | Captures fine acoustic details |
| Batch normalization | — | Stabilizes the branch output |
| Global average pooling | — | Summarizes average activation evidence |
| Global max pooling | — | Captures the strongest local acoustic evidence |
| Concatenation | average + max features | Produces the PCG embedding |

### Why use both global average and max pooling?

The two pooling operations provide complementary summaries:

- **Global average pooling** captures how broadly a pattern is present across the beat.
- **Global max pooling** captures whether a highly distinctive local event occurs anywhere in the beat.

Concatenating them produces a compact representation without flattening the entire temporal feature map.

## 8. Multimodal feature fusion

After the ECG and PCG branches independently encode their inputs, their embeddings are concatenated:

```text
fused = Concatenate([ECG_embedding, PCG_embedding])
```

This is **late feature fusion**. Each modality first learns a specialized representation, and the classifier then learns how to combine electrical and acoustic evidence for the final decision.

The fusion stage is:

| Layer | Configuration |
|---|---|
| Dense | 192 units, ReLU |
| Dropout | 0.30 |
| Dense | 96 units, ReLU |
| Dropout | 0.20 |
| Output | 1 unit, sigmoid |

The sigmoid output is interpreted as the probability of the abnormal class:

```text
p(abnormal) = sigmoid(logit)
```

A threshold of `0.5` is used to convert probability into a binary prediction.

## 9. Training configuration

The model is compiled with:

```text
Optimizer: Adam
Learning rate: 5e-5
Gradient clipping: clipnorm=1.0
Loss: binary cross-entropy
Batch size: 32
Maximum epochs: 40
Random seed: 42
```

The monitored metrics are:

- Accuracy
- Area under the ROC curve (AUC)
- Precision
- Recall

Two callbacks are used:

- **EarlyStopping** monitors validation loss with patience `8` and restores the best weights.
- **ReduceLROnPlateau** halves the learning rate after `3` epochs without validation-loss improvement, down to a minimum of `1e-6`.

## 10. Evaluation

After training, the notebook predicts probabilities on the held-out test set and computes:

- Accuracy
- F1 score
- ROC AUC
- Confusion matrix
- Precision and recall through a classification report

The notebook uses the following class names:

```text
0 → Normal
1 → Abnormal
```

The notebook output should be treated as an experiment log rather than a universally reproducible benchmark. Results can vary with library versions, dataset mirrors, hardware, random seeds, and preprocessing behavior.

## 11. Installation

The notebook installs or uses the following main packages:

```bash
pip install kaggle wfdb biosppy peakutils scipy numpy pandas tqdm tensorflow scikit-learn
```

A GPU is recommended for training. The notebook metadata indicates a Google Colab T4 GPU configuration, but CPU execution may also be possible with considerably longer training time.

## 12. Running the notebook

### Google Colab

1. Open `ML_PACFNET2.ipynb` in Google Colab.
2. Select a Python 3 runtime.
3. Enable a GPU runtime if available.
4. If the dataset is not already present, configure `kaggle.json`.
5. Run the cells from top to bottom.
6. Review the generated dataset checks, model summary, training history, and final metrics.

### Local Jupyter environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install jupyter kaggle wfdb biosppy peakutils scipy numpy pandas tqdm tensorflow scikit-learn
jupyter notebook
```

The notebook currently uses Colab-style paths such as `/content/...`. For local execution, update the dataset and generated `.npz` paths to locations available on your machine.

## 13. Important methodological considerations

### Beat-level versus patient/record-level splitting

The current notebook performs a random **beat-level** train/test split while retaining `record_ids` for traceability. Multiple beats from the same source record can therefore appear in both training and testing partitions. This can make evaluation optimistic because the model may encounter highly similar signal characteristics from the same record in both sets.

For a stronger generalization estimate, future experiments should split by `record_ids` before beat extraction or use a grouped split such as `GroupShuffleSplit`, ensuring that all beats from a record remain in one partition.

### Label granularity

The record-level normal/abnormal label is copied to every extracted beat from that record. This means the model is trained for beat-level prediction using record-level supervision. A beat from an abnormal record is not necessarily independently abnormal, so this limitation should be acknowledged when interpreting predictions.

### Clinical use

The model has not been validated for clinical deployment, prospective monitoring, diagnosis, or treatment decisions. External validation, patient-level evaluation, calibration, robustness analysis, and clinical review would be required before any real-world use.

## 14. Future improvements

Potential extensions include:

- patient- or record-level grouped cross-validation
- patient-level aggregation of beat predictions
- attention-based ECG–PCG fusion
- explicit ECG/PCG temporal alignment learning
- class-weighted or focal loss comparisons
- ablation studies for ECG-only, PCG-only, and fused models
- calibration and decision-threshold analysis
- explainability with saliency maps or integrated gradients
- reproducible environment files and saved model checkpoints
- external validation on an independent cardiac dataset

## Citation and dataset attribution

This project uses the PhysioNet/CinC Challenge 2016 heart-sound dataset through a Kaggle-accessible mirror. 
