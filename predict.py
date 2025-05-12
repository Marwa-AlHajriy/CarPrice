import numpy as np
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('used_car_price_model.pkl')
columns = pd.read_csv('x_nonvintage_columns.csv')['columns'].values

# Category options based on your cleaned data
states = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']
manufacturers = ['acura', 'alfa-romeo', 'audi', 'bmw', 'buick', 'cadillac', 'chevrolet', 'chrysler', 'dodge', 'fiat', 'ford', 'gmc', 'honda', 'hyundai', 'infiniti', 'jaguar', 'jeep', 'kia', 'lexus', 'lincoln', 'mazda', 'mercedes-benz', 'mercury', 'mini', 'mitsubishi', 'nissan', 'pontiac', 'porsche', 'ram', 'rover', 'saturn', 'subaru', 'tesla', 'toyota', 'volkswagen', 'volvo']
conditions = ['excellent', 'fair', 'good', 'like new', 'new']
fuels = ['diesel', 'electric', 'gas', 'hybrid']
transmissions = ['automatic', 'manual']
drives = ['4wd', 'fwd']
colors = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'purple', 'red', 'silver', 'white', 'yellow']

# Get user input
odometer = float(input("Enter odometer reading (e.g., 60000): "))
vehicle_age = float(input("Enter vehicle age in years (e.g., 5): "))

state = input(f"Enter state (2-letter code) from: {states}\n> ").lower()
manufacturer = input(f"Enter manufacturer from: {manufacturers}\n> ").lower()
condition = input(f"Enter condition from: {conditions}\n> ").lower()
fuel = input(f"Enter fuel type from: {fuels}\n> ").lower()
transmission = input(f"Enter transmission from: {transmissions}\n> ").lower()
paint_color = input(f"Enter paint color from: {colors}\n> ").lower()
drive = input(f"Enter drive type from: {drives}\n> ").lower()

# Build input vector
x = np.zeros(len(columns))
x[0] = odometer
x[1] = vehicle_age
for feature in [state, manufacturer, condition, fuel, transmission, paint_color, drive]:
    if feature in columns:
        idx = np.where(columns == feature)[0][0]
        x[idx] = 1

# Predict and display
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

input_df = pd.DataFrame([x], columns=columns)
predicted_price = model.predict(input_df)[0]
print(f"\nEstimated car price: ${predicted_price:.2f}")
