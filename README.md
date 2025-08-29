
# 2nd Hand Car Price Prediction

Analyze second-hand car prices from the [Craigslist Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) to predict the price of used vehicles.


## Steps

- **Data Cleaning**
  - Dropped irrelevant features and ones with a lot of NA
  - Removed motorcycles (`harley-davidson`)
  - Grouped manufacturers with small samples  into an `other` category
  - Created `vehicle_age` from `year` and `posting_date`
  - Removed outliers in `price` (IQR method per state)
  - Filtered unrealistic odometer readings
  - One-hot encoded categorical variables and dropped redundant levels

- **Data Splitting**
  - Separated into vintage and non-vintage vehicles
  - Trained only on non-vintage cars for modeling

- **Modeling**
  - Assesed multiple regressors: Linear, Lasso, Decision Tree, Random Forest, XGBoost
  - Tuned models using `GridSearchCV`
 

- **Evaluation**
  - Used R² score as metrics
  - Final model (Random Forest) achieved **~0.82 R²** on the test set

## Run the Model Locally in your Terminal 
You will be asked to provide car details after following the instructions:

**1. Clone the Repository**
```bash
git clone https://github.com/Marwa-AlHajriy/CarPrice.git
cd CarPrice
```
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
**3. Run prediction**
```bash
python predict.py
```
