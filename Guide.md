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

### Image Retrieval

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

## 2.3. LLM Usage

In this competition, each team will be provided with access to a Large Language Model (LLM) to help brainstorm ideas, implement code, and debug. However, there are a few constraints that should be considered:

- There will be a limit on the context window (usually around 2,000 tokens).
- The LLM is only available for the first 5 hours and will be turned off for the last 1 hour.

The first constraint shapes how we should use the LLM, while the second defines our time management strategy which will be discussed in the section below. We will focus only on the first constraint here.

In terms of brainstorming, if a task is unfamiliar, the LLM is a great way to kick things off. You can use it to identify the type of task, ask for recommended approaches and figure out what needs to be done to solve the problem. 

For coding, the context window limitation is a strict constraint. Because of it, you cannot ask the LLM to generate large code all at once. Therefore, it is much better to do things like:

1. Generate an outline for the end-to-end pipeline.
2. Generate code for each function/component individually (1 session for each).
3. Debug specific code snippets if you run into errors.

## 2.4. Version Control

Version control is recommended to track progress, document changes and experiments, and ensure result reproducibility. There are a few great methods to start with:

- **Naming Convention:** One recommended method is saving your notebook after each change and run using a naming format such as `<score>_<applied_techniques/hyperparameters>` (e.g., `0.842_DataAugmentation.ipynb`).
- **Sync Weights with Notebooks:** Always save the model checkpoints with the exact same name as the notebook to easily match the best weights for final inference (e.g., `0.842_DataAugmentation.pth` for `0.842_DataAugmentation.ipynb`).
- **Keep a Quick Log:** Maintain a brief note to track what you have tried and avoid duplicating experiments.

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
- Next, improve the architecture and training paradigm to achieve the best score (remember to use version control for better tracking and reproducibility).

**The last 1 hour:** Focus only on fine-tuning hyperparameters, which is the safest approach (in case you do not know how to code just like me). Additional data processing is also a good option, though only advanced techniques should be used here since everything essential should already be done within the first 5 hours.