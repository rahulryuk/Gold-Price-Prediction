from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("gld_price_data.csv")

# Preprocess the dataset
data['Date'] = pd.to_datetime(data['Date'])
data = data.drop_duplicates()
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

features = ['SPX', 'USO', 'SLV', 'EUR/USD', 'day', 'month', 'year']

target = 'GLD'

# Perform K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
r2_scores = []

for train_index, test_index in kf.split(data):
    train, test = data.iloc[train_index], data.iloc[test_index]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mae_scores.append(mae)
    r2_scores.append(r2)

average_mae = sum(mae_scores) / len(mae_scores)
average_r2 = sum(r2_scores) / len(r2_scores)

# Load the trained model and scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])
y = data[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Define routes (Flask app)
@app.route('/')
def index():
    return render_template('index.html',average_r2=average_r2,average_mae =average_mae )

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {}
    print(user_input)
    for feature in features:
        if feature in ['day', 'month', 'year']:
            # Process day, month, and year directly
            user_input[feature] = int(request.form[feature]) if request.form[feature] else 0
        else:
            # Process numeric features
            user_input[feature] = float(request.form[feature]) if request.form[feature] else 0.0

    user_df = pd.DataFrame([user_input])
    user_input_scaled = scaler.transform(user_df[features[0:]])  # Exclude 'day', 'month', 'year'
    user_prediction = model.predict(user_input_scaled)

    return render_template('index.html', prediction=user_prediction[0], average_r2=average_r2, average_mae=average_mae)

if __name__ == '__main__':
    app.run(debug=True)
