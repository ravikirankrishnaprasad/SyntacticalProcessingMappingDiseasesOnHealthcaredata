# Disease-Treatment Mapping Using Custom NER

This repository contains a machine learning project that focuses on building a custom Named Entity Recognition (NER) model to identify **Diseases** and **Treatments** from medical text data. The objective is to extract all predicted treatments corresponding to diseases using a Conditional Random Fields (CRF) model.

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
