import joblib
import pandas as pd

# Load Model
model = joblib.load("C:\\Users\\Raghul K\\OneDrive\\Desktop\\Git\\loan_prediction_model.pkl")

print("Enter Loan Applicant Details")

gender = int(input("Gender (Male=1 Female=0): "))
married = int(input("Married (Yes=1 No=0): "))
education = int(input("Graduate=1 Not Graduate=0: "))
self_emp = int(input("Self Employed (Yes=1 No=0): "))
income = float(input("Applicant Income: "))
co_income = float(input("Co Applicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Amount Term: "))
credit = float(input("Credit History (1 or 0): "))
property_area = int(input("Property Area (Urban=2 Semiurban=1 Rural=0): "))

sample = pd.DataFrame([[
    gender,
    married,
    education,
    self_emp,
    income,
    co_income,
    loan_amount,
    loan_term,
    credit,
    property_area
]], columns=[
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
])

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Rejected")