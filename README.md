# SMS Spam Detection

SMS Spam Detection is an end-to-end academic project by Ibrahim Irfan and Bilal Ahmed (University of Lahore) focused on classifying SMS messages as spam or not spam.

The repository contains:
- A Python training and experimentation script for text preprocessing, feature extraction, and model evaluation.
- An Android application that performs on-device inference using a TensorFlow Lite model.
- Supporting artifacts such as research papers and presentation material.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Android App Setup and Run](#android-app-setup-and-run)
- [Python Training Workflow](#python-training-workflow)
- [Model and Inference Details](#model-and-inference-details)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

## Project Overview
This project demonstrates an applied machine learning pipeline for SMS spam detection:
1. Collect and clean SMS data.
2. Convert text into numerical features.
3. Train and evaluate classification models.
4. Deploy the trained model to Android using TensorFlow Lite.
5. Run inference locally on-device with no network dependency.

The mobile app predicts two probabilities for a user-entered message:
- `SPAM`
- `NOT SPAM`

## Key Features
- SMS text preprocessing with stopword removal and normalization.
- TF-IDF-based feature extraction.
- Baseline model training/evaluation in Python.
- Android inference pipeline with:
  - local vocabulary loading (`word_dict.json`)
  - tokenization and sequence padding
  - TensorFlow Lite model execution (`model.tflite`)
- Simple user interface for real-time message classification.

## Repository Structure
```text
SMS-Spam-Detection/
|- README.md
|- Android Code/
|  |- app/
|  |  |- src/main/assets/
|  |  |  |- model.tflite
|  |  |  |- word_dict.json
|  |  |- src/main/java/com/ml/quaterion/spamo/
|  |  |  |- Classifier.kt
|  |  |  |- MainActivity.kt
|  |  |- src/main/res/layout/activity_main.xml
|  |- build.gradle
|  |- gradle/wrapper/gradle-wrapper.properties
|- Python Code/
|  |- Python Code/
|  |  |- spam.csv
|  |  |- spm.py
|- Presentations/
|- Research Papers/
```

## System Architecture
```text
SMS Input (User)
	-> Text normalization and lowercasing
	-> Tokenization using word dictionary
	-> Sequence padding to fixed length (171)
	-> TensorFlow Lite inference
	-> Output probabilities: [NOT SPAM, SPAM]
```

Training side (Python):
```text
CSV dataset
	-> Cleaning (HTML/regex/stopwords)
	-> TF-IDF vectorization
	-> Train/test split
	-> Classifier training and metrics
	-> Export artifacts for mobile inference
```

## Tech Stack

### Android App
- Kotlin `1.3.21`
- Android Gradle Plugin `3.4.0`
- Gradle `5.1.1`
- compile/target SDK `28`
- TensorFlow Lite `1.13.1`
- AndroidX AppCompat/Core/ConstraintLayout

### Python Training Script
- Python script based workflow (`spm.py`)
- NumPy, Pandas
- scikit-learn
- NLTK stopwords
- BeautifulSoup (bs4)
- imbalanced-learn (imported)
- Matplotlib (optional plotting)

## Android App Setup and Run

### Prerequisites
- Android Studio (recommended for this project structure)
- JDK compatible with legacy Android Gradle Plugin 3.4.0 (typically JDK 8)
- Android SDK 28 installed

### Steps
1. Open Android Studio.
2. Select **Open** and choose the `Android Code/` folder.
3. Let Gradle sync complete.
4. Build the app:
	- from IDE: **Build > Make Project**, or
	- terminal (inside `Android Code/`):

```bash
./gradlew assembleDebug
```

5. Run on emulator/device (min SDK 23).
6. Enter an SMS message and tap the classify button.

### Expected Behavior
- The app loads `word_dict.json` from assets.
- Message is tokenized and padded to length `171`.
- `model.tflite` runs inference via TensorFlow Lite Interpreter.
- UI shows probabilities for spam and not-spam classes.

## Python Training Workflow

### Prerequisites
- Python environment with required packages installed.
- Dataset file available at `Python Code/Python Code/spam.csv`.

### Install Dependencies
Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn beautifulsoup4 nltk imbalanced-learn matplotlib
python -c "import nltk; nltk.download('stopwords')"
```

### Run the Training Script
From `Python Code/Python Code/`:

```bash
python spm.py
```

The script:
- Reads the dataset (`spam.csv`).
- Cleans text (HTML removal, non-letter filtering, lowercasing, stopword filtering).
- Builds TF-IDF features (`max_features=1000`).
- Splits data into train/test (80/20).
- Trains a Gaussian Naive Bayes model.
- Prints classification report and final accuracy.

## Model and Inference Details
- Input text is split by spaces and mapped via vocabulary indices.
- Unknown tokens are mapped to index `0`.
- Sequence is padded/truncated to fixed length `171`.
- Inference output is a float array with two class scores.

Important implementation note:
- Ensure model training preprocessing, token mapping, and sequence length are consistent with mobile preprocessing; mismatch can degrade prediction quality.

## Testing

### Android
Inside `Android Code/`:

```bash
./gradlew test
./gradlew connectedAndroidTest
```

### Python
No dedicated automated test suite is currently included for the training script. Model quality is observed from printed classification metrics.

## Troubleshooting

### Gradle sync/build issues
- This project uses older Android/Kotlin/Gradle versions.
- If build fails on modern toolchains, use the matching legacy environment first, then migrate versions incrementally.

### TensorFlow Lite model load errors
- Confirm `model.tflite` exists in `app/src/main/assets/`.
- Verify `aaptOptions { noCompress "tflite" }` remains enabled.

### Vocabulary/tokenization mismatch
- Ensure `word_dict.json` corresponds to the model that is currently bundled.
- Keep `setMaxLength(171)` aligned with model training expectations.

### NLTK stopwords errors in Python
- Download stopwords before running:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Limitations
- Current Android stack is pinned to legacy dependencies.
- Python script is a single-file workflow and not yet modularized.
- No CI pipeline or reproducible training/export pipeline is included.
- No benchmark suite on diverse, real-world SMS distributions.

## Future Improvements
- Upgrade Android project to newer Gradle, AGP, Kotlin, and TensorFlow Lite versions.
- Refactor Python training into modular scripts/notebooks with reproducible config.
- Add model export pipeline and artifact versioning.
- Add unit/integration tests for preprocessing and inference.
- Improve UI/UX with confidence labels and explainability hints.

## Contributors
- Ibrahim Irfan
- Bilal Ahmed

## License
The Android project includes a `LICENSE.txt`. If you intend to open-source the entire repository, add a top-level license file and update this section accordingly.
