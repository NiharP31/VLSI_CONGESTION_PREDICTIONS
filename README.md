# VLSI Congestion Prediction using CircuitNet Dataset

## Overview

This project aims to predict congestion in Very Large Scale Integration (VLSI) circuits using machine learning techniques. It leverages the CircuitNet dataset to train and evaluate models for accurate congestion prediction, which is crucial for optimizing chip design and improving overall performance.

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
```

## Usage

The entire pipeline is orchestrated through the main.py file. To run the project:
```bash
python src/main.py
```
This command executes the following steps in sequence:
1. Data analysis and preprocessing
2. Dataset preparation
3. Model initialization
4. Training
5. Evaluation
6. Results visualization
You can modify the main.py file to adjust parameters or toggle specific parts of the pipeline.

## Model Architecture

The project uses a custom neural network architecture designed specifically for VLSI congestion prediction. Key components include:
- Positional encoding to capture spatial relationships
- Multi-layer perceptron (MLP) layers for feature extraction
- Attention mechanisms to focus on relevant circuit areas
For more details, refer to src/model.py.

## Results

After running the pipeline, you can find the results in the console output. The model's performance metrics include:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²) Score
Visualizations of the results will be generated and saved in a designated output directory.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- CircuitNet Dataset


