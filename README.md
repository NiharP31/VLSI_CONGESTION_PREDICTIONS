# VLSI Congestion Prediction using CircuitNet Dataset

## Overview

This project aims to predict congestion in Very Large Scale Integration (VLSI) circuits using machine learning techniques. It leverages the CircuitNet dataset to train and evaluate models for accurate congestion prediction, which is crucial for optimizing chip design and improving overall performance.

## Project Structure

VLSI_congestion_Prediction/  
│  
├── data/  
│ └── pin_positions/  
│  
└── src/  
├── data_analysis.py  
├── dataset.py  
├── evaluation.py  
├── model.py  
├── main.py  
├── positional_encoding.py  
├── train.py  
└── visualization.py  


## Features

- Data analysis and preprocessing of the CircuitNet dataset
- Custom dataset implementation for efficient data loading
- Advanced model architecture for congestion prediction
- Positional encoding to capture spatial information
- Comprehensive training pipeline
- Evaluation metrics for model performance
- Visualization tools for results and insights

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn

To install the required packages, run:

```bash
pip install -r requirements.txt

