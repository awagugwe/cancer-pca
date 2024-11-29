# Cancer Dataset PCA Analysis

## Overview
This project implements Principal Component Analysis (PCA) on the breast cancer dataset from scikit-learn to identify essential variables for donor funding at the Anderson Cancer Center. The analysis includes dimensionality reduction and optional logistic regression for prediction.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Analysis Components](#analysis-components)
- [Results Interpretation](#results-interpretation)
- [Contributing](#contributing)

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/awagugwe/cancer-pca-analysis.git
cd cancer-pca-analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Run the main analysis script:
```bash
python cancer_pca_analysis.py
```

2. The script will generate:
   - PCA transformation of the dataset
   - Visualizations of the results
   - Feature importance analysis
   - (Optional) Logistic regression predictions

## Project Structure
```
cancer-pca-analysis/
│
├── cancer_pca_analysis.py    # Main analysis script
├── requirements.txt          # Package dependencies
├── README.md                # This file
└── visualizations/          # Generated plots and figures
```

## Analysis Components

### 1. PCA Implementation
- Loads breast cancer dataset
- Standardizes features
- Applies PCA transformation
- Calculates explained variance ratios

### 2. Dimensionality Reduction
- Reduces 30 features to 2 PCA components
- Generates visualization of:
  - Explained variance ratio
  - PCA scatter plot
  - Feature importance

### 3. Logistic Regression (Optional)
- Implements prediction on PCA-transformed data
- Provides accuracy metrics
- Generates classification report

## Results Interpretation

### PCA Analysis
The script provides:
1. Explained variance ratios for each principal component
2. Visualization of cumulative explained variance
3. Scatter plot of samples in PCA space
4. Feature importance in principal components

### Logistic Regression
If implemented, the results include:
1. Model accuracy
2. Detailed classification report
3. Performance metrics

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Scikit-learn for providing the breast cancer dataset
- Anderson Cancer Center for the project requirements

## Contact
For questions or feedback, please open an issue in the GitHub repository.

---
*Note: This project is part of the Anderson Cancer Center data analysis initiative.*
