from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


# Load the complete ML pipeline
# The pipeline contains:
# - StandardScaler
# - OneHotEncoder
# - RandomForestRegressor
model = joblib.load("car_price_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get data from HTML form
    data = {
        "symboling": float(request.form["symboling"]),

        # Categorical features
        "fueltype": request.form["fueltype"],
        "aspiration": request.form["aspiration"],
        "doornumber": request.form["doornumber"],
        "carbody": request.form["carbody"],
        "drivewheel": request.form["drivewheel"],
        "enginelocation": request.form["enginelocation"],

        # Numerical features
        "wheelbase": float(request.form["wheelbase"]),
        "carlength": float(request.form["carlength"]),
        "carwidth": float(request.form["carwidth"]),
        "carheight": float(request.form["carheight"]),
        "curbweight": float(request.form["curbweight"]),

        # Categorical features
        "enginetype": request.form["enginetype"],
        "cylindernumber": request.form["cylindernumber"],

        # Numerical features
        "enginesize": float(request.form["enginesize"]),

        # Categorical feature
        "fuelsystem": request.form["fuelsystem"],

        # Numerical features
        "boreratio": float(request.form["boreratio"]),
        "stroke": float(request.form["stroke"]),
        "compressionratio": float(request.form["compressionratio"]),
        "horsepower": float(request.form["horsepower"]),
        "peakrpm": float(request.form["peakrpm"]),
        "citympg": float(request.form["citympg"]),
        "highwaympg": float(request.form["highwaympg"])
    }


    # Convert the dictionary into a Pandas DataFrame
    #
    # The DataFrame is important because the pipeline
    # was trained using named columns.
    input_data = pd.DataFrame([data])


    # Make prediction
    #
    # We DON'T manually scale or encode the data here.
    # The saved pipeline does that automatically.
    prediction = model.predict(input_data)[0]


    # Send prediction back to HTML
    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)