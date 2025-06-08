# ❤️‍🔥 Heart Disease Detector

This project predicts the likelihood of heart disease using a machine learning model trained on the [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) dataset.

## Project Structure

- `heart_disease_prediction.ipynb` — Jupyter Notebook version (recommended for exploration and step-by-step explanation)
- `model.py` — Python script for training the model
- `main.py` — Python script for making predictions with the trained model
- `heartrisk_detector_model.keras` — Saved Keras model
- `heartdisease2020_dataset_by_kamil/heart_2020_cleaned.csv` — Dataset 
- `requirements.txt` — List of dependencies

## Setup

1. **Clone the repository**

2. **Create and activate a virtual environment**
    ```sh
    virtualenv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the dataset**
    - Download `heart_2020_cleaned.csv` from [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
    - Place it in the `heartdisease2020_dataset_by_kamil/` directory

## Usage

### Jupyter Notebook Version

Open `heart_disease_prediction.ipynb` in JupyterLab or VS Code and run the cells step by step. This version includes explanations, visualizations, and interactive exploration.

### Python Script Version

#### 1. Train the model

```sh
python model.py
```
- This will train the model and save it as `heartrisk_detector_model.keras`.

#### 2. Make predictions

```sh
python main.py
```
- This will load the saved model and make predictions on sample data.

## Notes

- The dataset are **not included** in the repository. Add them to `.gitignore` to avoid pushing large files.
- Both versions use the same data preprocessing and model architecture.

## Requirements

See [requirements.txt](requirements.txt) for the full list. Main dependencies:
- pandas
- scikit-learn
- tensorflow

---

Feel free to use either the notebook or the Python scripts depending on your workflow!