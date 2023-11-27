import pickle

import firebase_admin
import numpy as np
import pandas as pd
import pyrebase
from firebase_admin import auth, credentials, firestore
from flask import Flask, flash, redirect, render_template, request, session
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)
app.secret_key = "secret"

# Load the ARIMA models
with open("./ARIMA/ARIMA_VP.pkl", "rb") as f:
    model_vp = pickle.load(f)

with open("./ARIMA/ARIMA_AH.pkl", "rb") as f:
    model_ah = pickle.load(f)

with open("./ARIMA/ARIMA_FP.pkl", "rb") as f:
    model_fp = pickle.load(f)

# Initialize Firebase app
config = {
    "apiKey": "AIzaSyBixtA4v5mvxKvaTU61iq9Fr2Ln2OWlf3o",
    "authDomain": "tomatocare-78e23.firebaseapp.com",
    "projectId": "tomatocare-78e23",
    "storageBucket": "tomatocare-78e23.appspot.com",
    "messagingSenderId": "437959910172",
    "appId": "1:437959910172:web:dab8c80225929289dd90d9",
    "databaseURL": "",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# Initialize the app with the service account credentials
cred = credentials.Certificate(
    "tomatocare-78e23-firebase-adminsdk-by348-b01efacff5.json"
)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Routes


# Define the route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Define the route for the forecasting page
@app.route("/forecasting", methods=["GET", "POST"])
def forecasting():
    if request.method == "POST":
        # Retrieve user input from the form
        num_years = int(request.form["num_years"])
        selected_option = request.form["selected_option"]
        number = request.form.get("num_years")

        if selected_option == "AreaHarvested":
            # Step 1: Load the data
            data = pd.read_csv("ARIMA/csv/AreaHarvested.csv")

            # Convert 'Year' column to string type
            data["Year"] = data["Year"].astype(str)
            data["YearQuarter"] = data["Year"] + "-" + data["TimePeriod"]

            data["AreaHarvested_log"] = np.log(data["AreaHarvested"])
            train_data = data["AreaHarvested_log"].iloc[: int(len(data) * 0.7)]

            # Step 3: Fit the ARIMA model
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 4: Make time series predictions
            test_data = data["AreaHarvested_log"].iloc[int(len(data) * 0.7) :]
            forecast = model_fit.forecast(steps=len(test_data))

            # Step 2: Fit the ARIMA modelg
            model = ARIMA(data["AreaHarvested_log"], order=(4, 1, 0))
            model_fit = model.fit()

            combined_data = pd.concat(
                [data[["YearQuarter", "AreaHarvested_log"]], pd.Series(forecast)],
                axis=1,
            )
            combined_data.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_data = data["AreaHarvested"].iloc[: int(len(data) * 0.7)]
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 3: Make time series predictions
            last_year = int(data["Year"].iloc[-1])
            last_year = last_year + 1
            future_years = pd.date_range(
                start=f"{last_year}-01-01", periods=num_years * 4, freq="Q"
            )
            forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

            # Step 4: Create a DataFrame with the predicted data
            prediction_df = pd.DataFrame(
                {
                    "Year": future_years.year,
                    "TimePeriod": future_years.quarter,
                    "Actual": data["AreaHarvested"].values[-num_years * 4 :],
                    "Forecast": forecast,
                }
            )

            # Step 9: Get percentage change
            percent_changes = []
            start_index = 0
            for i in range(4):
                end_index = -(4 - i)
                pd_change = (
                    prediction_df["Forecast"].iloc[end_index]
                    - prediction_df["Forecast"].iloc[start_index]
                ) / prediction_df["Forecast"].iloc[start_index]
                percent_changes.append(round(pd_change * 100, 2))
                start_index += 1

            # Calculate average percentage change
            average_percent_change = sum(percent_changes) / len(percent_changes)

            # Format percent changes for display
            formatted_changes = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_df["Year"].iloc[start_index - 1]
                year_end = prediction_df["Year"].iloc[end_index]
                formatted_changes.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changes[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_areaResults.html",
                prediction_df=prediction_df.to_dict(orient="records"),
                number=number,
                formatted_changes=formatted_changes,
                average_percent_change=average_percent_change,
            )

        elif selected_option == "FarmgatePrices":
            # Step 1: Load the data
            data = pd.read_csv("ARIMA/csv/FarmgatePrices.csv")

            # To avoid using scientific notation
            pd.set_option("display.float_format", lambda x: "%d" % x)

            # Convert 'Year' column to string type
            data["Year"] = data["Year"].astype(str)
            data["YearQuarter"] = data["Year"] + "-" + data["TimePeriod"]

            data["FarmgatePrices"] = np.log(data["FarmgatePrices"])
            train_data = data["FarmgatePrices"].iloc[: int(len(data) * 0.7)]

            # Step 3: Fit the ARIMA model
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 4: Make time series predictions
            test_data = data["FarmgatePrices"].iloc[int(len(data) * 0.7) :]
            forecast = model_fit.forecast(steps=len(test_data))

            # Step 2: Fit the ARIMA model
            model = ARIMA(data["FarmgatePrices"], order=(4, 1, 0))
            model_fit = model.fit()

            combined_data = pd.concat(
                [data[["YearQuarter", "FarmgatePrices"]], pd.Series(forecast)],
                axis=1,
            )
            combined_data.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_data = data["FarmgatePrices"].iloc[: int(len(data) * 0.7)]
            model = ARIMA(train_data, order=(4, 1, 0))
            model_fit = model.fit()

            # Step 3: Make time series predictions
            last_year = int(data["Year"].iloc[-1])
            last_year = last_year + 1
            future_years = pd.date_range(
                start=f"{last_year}-01-01", periods=num_years * 4, freq="Q"
            )
            forecast = pd.Series(model_fit.forecast(steps=num_years * 4).values)

            # Step 4: Create a DataFrame with the predicted data
            prediction_df = pd.DataFrame(
                {
                    "Year": future_years.year,
                    "TimePeriod": future_years.quarter,
                    "Actual": data["FarmgatePrices"].values[-num_years * 4 :],
                    "Forecast": forecast,
                }
            )

            # Step 9: Get percentage change
            percent_changes = []
            start_index = 0
            for i in range(4):
                end_index = -(4 - i)
                pd_change = (
                    prediction_df["Forecast"].iloc[end_index]
                    - prediction_df["Forecast"].iloc[start_index]
                ) / prediction_df["Forecast"].iloc[start_index]
                percent_changes.append(round(pd_change * 100, 2))
                start_index += 1

            # Calculate average percentage change
            average_percent_change = sum(percent_changes) / len(percent_changes)

            # Format percent changes for display
            formatted_changes = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_df["Year"].iloc[start_index - 1]
                year_end = prediction_df["Year"].iloc[end_index]
                formatted_changes.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changes[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_farmgateResult.html",
                prediction_df=prediction_df.to_dict(orient="records"),
                number=number,
                formatted_changes=formatted_changes,
                average_percent_change=average_percent_change,
            )

        elif selected_option == "VolumeDemandAndProduction":
            # Step 1: Load the data
            dataDemand = pd.read_csv("ARIMA/csv/VolumeDemand.csv")
            dataProduction = pd.read_csv("ARIMA/csv/VolumeProduction.csv")

            # To avoid using scientific notation
            pd.set_option("display.float_format", lambda x: "%d" % x)

            # Convert 'Year' column to string type
            dataDemand["Year"] = dataDemand["Year"].astype(str)
            dataDemand["YearQuarter"] = (
                dataDemand["Year"] + "-" + dataDemand["TimePeriod"]
            )
            dataProduction["Year"] = dataProduction["Year"].astype(str)
            dataProduction["YearQuarter"] = (
                dataProduction["Year"] + "-" + dataProduction["TimePeriod"]
            )

            dataDemand["VolumeDemand"] = np.log(dataDemand["VolumeDemand"])
            train_dataDemand = dataDemand["VolumeDemand"].iloc[
                : int(len(dataDemand) * 0.7)
            ]

            dataProduction["VolumeProduction_log"] = np.log(
                dataProduction["VolumeProduction"]
            )
            train_dataProduction = dataProduction["VolumeProduction_log"].iloc[
                : int(len(dataProduction) * 0.7)
            ]

            # Step 3: Fit the ARIMA model
            modelDemand = ARIMA(train_dataDemand, order=(4, 1, 0))
            model_fitDemand = modelDemand.fit()
            modelProduction = ARIMA(train_dataProduction, order=(4, 1, 0))
            model_fitProduction = modelProduction.fit()

            # Step 4: Make time series predictions
            test_dataDemand = dataDemand["VolumeDemand"].iloc[
                int(len(dataDemand) * 0.7) :
            ]
            forecastDemand = model_fitDemand.forecast(steps=len(test_dataDemand))
            test_dataProduction = dataProduction["VolumeProduction_log"].iloc[
                int(len(dataProduction) * 0.7) :
            ]
            forecastProduction = model_fitProduction.forecast(
                steps=len(test_dataProduction)
            )

            # Step 2: Fit the ARIMA model graph
            modelDemand = ARIMA(dataDemand["VolumeDemand"], order=(4, 1, 0))
            model_fitDemand = modelDemand.fit()

            modelProduction = ARIMA(
                dataProduction["VolumeProduction_log"], order=(4, 1, 0)
            )
            model_fitProduction = modelProduction.fit()

            combined_dataDemand = pd.concat(
                [
                    dataDemand[["YearQuarter", "VolumeDemand"]],
                    pd.Series(forecastDemand),
                ],
                axis=1,
            )
            combined_dataDemand.columns = ["Year", "Actual", "Forecast"]

            combined_dataProduction = pd.concat(
                [
                    dataProduction[["YearQuarter", "VolumeProduction_log"]],
                    pd.Series(forecastProduction),
                ],
                axis=1,
            )
            combined_dataProduction.columns = ["Year", "Actual", "Forecast"]

            # Fit the ARIMA model using the actual series
            train_dataDemand = dataDemand["VolumeDemand"].iloc[
                : int(len(dataDemand) * 0.7)
            ]
            modelDemand = ARIMA(train_dataDemand, order=(4, 1, 0))
            model_fitDemand = modelDemand.fit()

            train_dataProduction = dataProduction["VolumeProduction"].iloc[
                : int(len(dataProduction) * 0.7)
            ]
            modelProduction = ARIMA(train_dataProduction, order=(4, 1, 0))
            model_fitProduction = modelProduction.fit()

            # Step 3: Make time series predictions
            last_yearDemand = int(dataDemand["Year"].iloc[-1])
            last_yearDemand = last_yearDemand + 1
            future_yearsDemand = pd.date_range(
                start=f"{last_yearDemand}-01-01", periods=num_years * 4, freq="Q"
            )
            forecastDemand = pd.Series(
                model_fitDemand.forecast(steps=num_years * 4).values
            )

            last_yearProduction = int(dataProduction["Year"].iloc[-1])
            last_yearProduction = last_yearProduction + 1
            future_yearsProduction = pd.date_range(
                start=f"{last_yearProduction}-01-01", periods=num_years * 4, freq="Q"
            )
            forecastProduction = pd.Series(
                model_fitProduction.forecast(steps=num_years * 4).values
            )

            # Step 4: Create a DataFrame with the predicted data
            prediction_dfDemand = pd.DataFrame(
                {
                    "Year": future_yearsDemand.year,
                    "TimePeriod": future_yearsDemand.quarter,
                    "Actual": dataDemand["VolumeDemand"].values[-num_years * 4 :],
                    "Forecast": forecastDemand,
                }
            )

            prediction_dfProduction = pd.DataFrame(
                {
                    "Year": future_yearsProduction.year,
                    "TimePeriod": future_yearsProduction.quarter,
                    "Actual": dataProduction["VolumeProduction"].values[
                        -num_years * 4 :
                    ],
                    "Forecast": forecastProduction,
                }
            )

            # Step 9: Get percentage change
            percent_changesDemand = []
            start_indexDemand = 0
            for i in range(4):
                end_indexDemand = -(4 - i)
                pd_changeDemand = (
                    prediction_dfDemand["Forecast"].iloc[end_indexDemand]
                    - prediction_dfDemand["Forecast"].iloc[start_indexDemand]
                ) / prediction_dfDemand["Forecast"].iloc[start_indexDemand]
                percent_changesDemand.append(round(pd_changeDemand * 100, 2))
                start_indexDemand += 1

            percent_changesProduction = []
            start_indexProduction = 0
            for i in range(4):
                end_indexProduction = -(4 - i)
                pd_changeProduction = (
                    prediction_dfProduction["Forecast"].iloc[end_indexProduction]
                    - prediction_dfProduction["Forecast"].iloc[start_indexProduction]
                ) / prediction_dfProduction["Forecast"].iloc[start_indexProduction]
                percent_changesProduction.append(round(pd_changeProduction * 100, 2))
                start_indexProduction += 1

            # Calculate average percentage change
            average_percent_changeDemand = sum(percent_changesDemand) / len(
                percent_changesDemand
            )

            average_percent_changeProduction = sum(percent_changesProduction) / len(
                percent_changesProduction
            )

            # Format percent changes for display
            formatted_changesDemand = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_dfDemand["Year"].iloc[start_indexDemand - 1]
                year_end = prediction_dfDemand["Year"].iloc[end_indexDemand]
                formatted_changesDemand.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changesDemand[i]:.2f}%"
                )

            formatted_changesProduction = []
            for i in range(4):
                quarter = i + 1
                year_start = prediction_dfProduction["Year"].iloc[
                    start_indexProduction - 1
                ]
                year_end = prediction_dfProduction["Year"].iloc[end_indexProduction]
                formatted_changesProduction.append(
                    f"Q{quarter} {year_start}-{year_end}: {percent_changesProduction[i]:.2f}%"
                )

            # Render the forecasting_results.html template with the predicted data
            return render_template(
                "forecasting_volumeDemandAndProductionResult.html",
                prediction_dfDemand=prediction_dfDemand.to_dict(orient="records"),
                prediction_dfProduction=prediction_dfProduction.to_dict(
                    orient="records"
                ),
                number=number,
                formatted_changesDemand=formatted_changesDemand,
                formatted_changesProduction=formatted_changesProduction,
                average_percent_changeDemand=average_percent_changeDemand,
                average_percent_changeProduction=average_percent_changeProduction,
            )

    # Render the forecasting.html template for user input
    return render_template("forecasting.html")


# User login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Handle login form submission
        email = request.form["email"]
        password = request.form["password"]

        # Check if email or password is empty
        if not email or not password:
            flash("Please enter your email and password", "error")
            return render_template("login.html")

        try:
            # Sign in the user with email and password
            user = auth.sign_in_with_email_and_password(email, password)
            # Store user session data
            session["user"] = user
            return redirect("/")
        except Exception as e:
            error_message = e.strerror
            if "INVALID_PASSWORD" in error_message:
                flash(
                    "Invalid email or password. Please double-check your credentials and try again.",
                    "error",
                )
            elif "EMAIL_NOT_FOUND" in error_message:
                flash(
                    "Invalid email or password. Please double-check your credentials and try again.",
                    "error",
                )
            else:
                flash("An error occurred", "error")
            return render_template("login.html")
    else:
        # Display login form
        return render_template("login.html")


@app.route("/logout")
def logout():
    # Clear user session data
    session.pop("user", None)
    return redirect("/login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # Handle signup form submission
        email = request.form["email"]
        password = request.form["password"]
        username = request.form["username"]

        # Check if email, password, and username is empty
        if not email or not password or not username:
            flash("Please enter your email, password, and username", "error")
            return render_template("signup.html")

        # Create a new user in Firebase Authentication
        user = auth.create_user_with_email_and_password(email, password)

        # Store user details in Firestore database
        db.collection("users").document(user["localId"]).set(
            {"email": email, "name": username, "emailVerified": False}
        )

        # Send email verification link to the user
        auth.send_email_verification(user["idToken"])

        return redirect("/login")
    else:
        return render_template("signup.html")


@app.route("/forgotpassword", methods=["GET", "POST"])
def forgotpassword():
    if request.method == "POST":
        email = request.form["email"]
        if not email:
            flash("Please enter your email address and password", "error")
            return render_template("pseudo.html")
        try:
            auth.send_password_reset_email(email)
            flash(
                "An email has been sent with instructions for resetting your password.\nThis email may take a few minutes to arrive in your inbox.",
                "success",
            )
            return redirect("/login")
        except Exception:
            flash(
                "An email has been sent with instructions for resetting your password.\nThis email may take a few minutes to arrive in your inbox.",
                "success",
            )
            print(
                "An exception occurred, email not found in database"
            )  # Print error message
            return redirect("/login")
    return render_template("pseudo.html")


@app.route("/profile")
def user_profile():
    if "user" in session:
        user = session["user"]
        # Fetch user profile data from Firestore based on user ID
        profile = db.collection("users").document(user["localId"]).get().to_dict()
        return render_template("profile.html", profile=profile)
    else:
        return redirect("/login")


if __name__ == "__main__":
    app.run()
