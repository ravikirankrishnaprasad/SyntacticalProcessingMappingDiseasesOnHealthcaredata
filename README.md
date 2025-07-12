# Disease-Treatment Mapping Using Custom NER

This repository contains a machine learning project that focuses on building a custom Named Entity Recognition (NER) model to identify **Diseases** and **Treatments** from medical text data. The objective is to extract all predicted treatments corresponding to diseases using a Conditional Random Fields (CRF) model.

## 🧠 Problem Statement

This project simulates a real-world scenario in a health tech company called **BeHealthy**, which provides online consultations, e-prescriptions, and appointment management. The company generates a large volume of unstructured medical text daily, including doctor notes, therapy reviews, and consultation records.

The goal is to develop a **custom Named Entity Recognition (NER)** system capable of extracting **disease-treatment pairs** from medical text. For example, from a sentence like:

> *"The patient was a 62-year-old man with squamous cell lung cancer, which was first successfully treated by a combination of radiation therapy and chemotherapy."*

The system should identify:
```python
{'cancer': 'chemotherapy'}
```

However, these relationships are **not explicitly labeled** in the text. You'll build a model using **Conditional Random Fields (CRF)** to classify each word as either:
- `D`: Disease
- `T`: Treatment
- `O`: Other

---

## 📦 Dataset Overview

The dataset includes:

- `train_sent`, `test_sent`: Lists of words for each sentence, with one word per line
- `train_label`, `test_label`: Corresponding labels (`O`, `D`, `T`) for each word

Each sentence is separated by a blank line. You'll need to preprocess this format to convert it into complete sentences and their label sequences.

---

## 🎯 Project Objective

- Preprocess and format word- and label-level data into sentence form
- Extract features for each word (e.g., part-of-speech, capitalization, prefixes/suffixes)
- Train a **CRF model** to label diseases and treatments
- Evaluate the model on a test set
- Create a dictionary mapping **diseases to their likely treatments**

---

## 🛠️ Technologies Used

- Python
- scikit-learn
- sklearn-crfsuite
- NLTK
- Jupyter Notebook

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to identify diseases and treatments from unstructured medical text. The project involves:
- **Data Preprocessing**: Cleaning and transforming the raw medical text into structured sentences and corresponding labels.
- **Feature Engineering**: Creating relevant features for each word, such as word identity, capitalization, and context-based features (e.g., neighboring words).
- **Model Building**: Training a CRF model to identify disease (D) and treatment (T) labels in the dataset.
- **Evaluation**: Predicting the labels for the test dataset and extracting the predicted treatments corresponding to each disease.

## Dataset
The dataset consists of medical text, where:
- Each word is represented on a separate line.
- Sentences are separated by blank lines.
- Labels include:
  - `O` (Other)
  - `D` (Disease)
  - `T` (Treatment)

The provided dataset includes:
- `train_sent`: Training sentences.
- `train_label`: Labels corresponding to the training sentences.
- `test_sent`: Test sentences.
- `test_label`: Labels corresponding to the test sentences.

## Methodology
The project is divided into several steps:
1. **Data Preprocessing**: 
   - Construct complete sentences from individual words.
   - Create matching label sequences for each sentence.
   
2. **Feature Engineering**: 
   - Define features for each word, including:
     - Word identity (lowercased)
     - Suffixes (last 2-3 characters)
     - Capitalization check
     - Word is numeric or not
     - Contextual features (previous and next word characteristics)
     
3. **Model Building**:
   - A Conditional Random Fields (CRF) model is trained using the `sklearn_crfsuite` library.
   
4. **Disease-Treatment Extraction**: 
   - Predicted labels from the CRF model are used to extract the corresponding treatments for each identified disease.
   - This is stored in a structured format (e.g., pandas DataFrame).

## Installation

To run this project, you'll need the following dependencies:

```bash
pip install pandas numpy sklearn spacy sklearn-crfsuite
python -m spacy download en_core_web_sm
