
# 2nd Hand Car Price Prediction (US market)

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

## Deployment  

Deployed the model as a **Flask web app** using **Google Cloud Run**.  

Steps:  
1. Created a Flask API (`app.py`) to handle requests and predictions.  
2. Uploaded the trained model (`used_car_price_model.pkl`) to GitHub releases due to large file. 
3. Containerized the app using Docker `Dockerfile`.  
4. Deployed with **Google Cloud Run**.  
5. Created public **HTTPS endpoint**.  

---

## Run the site
[Car Price Prediction App](https://carprice-1017464960956.europe-west2.run.app/)  

