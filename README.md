**California House Price Predictor**
A machine learning model that predicts median house values across California districts using a Random Forest Regressor, trained on the California Housing dataset.

**How It Works**
The script operates in two modes automatically:

Train mode — if model.pkl does not exist, it trains the model on housing.csv, saves the model and pipeline, and writes a sample input.csv from the test split.
Inference mode — if model.pkl already exists, it loads the saved model and pipeline, runs predictions on input.csv, and writes results to output.csv.

**Dataset**

File: housing.csv
Target: median_house_value
Split: Stratified 80/20 train-test split based on income category

**Pipeline**
Step                                  Details
Imputation                Median strategy for numerical features
ScalingStandardScaler     for numerical features
Encoding                  OneHotEncoder for ocean_proximity
Model                     RandomForestRegressor(random_state=42)

**Project Structure**
├── housing.csv         # Raw dataset (required)
├── main.py             # Train and inference script
├── model.pkl           # Saved model (auto-generated)
├── pipeline.pkl        # Saved preprocessing pipeline (auto-generated)
├── input.csv           # Test split input (auto-generated)
├── output.csv          # Predictions output (auto-generated)
└── requirements.txt

**Tech Stack**
pandas · NumPy · scikit-learn
