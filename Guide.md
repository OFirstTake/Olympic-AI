# Olympic AI Guide

# 1. Foundational Task

## 1.1. Natural Language Processing

### Feature Extraction

### Fill-Mask

### Text Extraction

### Token Classification

### Setence Similarity

### Text Ranking

### Question Answering

### Text Generation

## 1.2. Computer Vision

### Image Feature Extraction

### Image Classification

### Object Detection

### Image Segmentation

### Keypoint Detection

### Depth Estimation

### Video Understanding

### Image-to-Text

## 1.3. Multimodal

### Visual Document Retrieval

### Document Question Answering

### Visual Question Answering

# 2. General Tips

## 2.1. Data

## 2.2. Trick & Technique

## 2.3. LLM Usage

## 2.5. Strategy 

### What to Do First

1. Read the rules.
2. Organize the final submission folder and ensure the reproducibility of the results. If model weights are required, make sure to save your model weights after training and load them when running inference on the test set.
3. Determine the list of provided pretrained models.

### Device Distribution & Time Management

Each team will be provided with 2 computers and 6 hours to train the best model. Therefore, a strategy to maximize the usage of both computers and time is necessary. Ideally, we suggest following this workflow:

**First 5 hours:**

- Dedicate 1 computer to data analysis and processing, while using the other computer to analyze the baseline model and results.
- After completing data processing, train the baseline model to see if the results improve, which indicates that the data has been effectively processed.
- Next, improve the architecture and training paradigm to achieve the best score.

**The last 1 hour:** Focus only on fine-tuning hyperparameters, which is the safest approach (in case you do not know how to code just like me). Additional data processing is also a good option, though only advanced techniques should be used here since everything essential should already be done within the first 5 hours.