# MLOps Assignment 4

This project demonstrates a simple Continuous Integration (CI) pipeline for a machine learning workflow using GitHub Actions.

## Project Components

- **train.py**: TensorFlow script for a GAN model trained on the MNIST dataset.
- **requirements.txt**: Python dependencies required to run the project.
- **Dockerfile**: Container configuration for running the training environment.
- **GitHub Actions Workflow**: Automates validation and testing of the ML environment.

## CI Pipeline Steps

The GitHub Actions workflow performs the following tasks:

1. Set up Python 3.10
2. Install project dependencies
3. Run a linter check using flake8
4. Perform a dry test of the ML environment
5. Upload the README.md file as a workflow artifact named **project-doc**

## Purpose

The purpose of this assignment is to demonstrate how CI pipelines can automatically validate machine learning projects and ensure reproducibility.
