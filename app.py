from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load both models
logistic_model = joblib.load('credit_model.pkl')
tree_model = joblib.load('credit_model_tree.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    form_data = request.form
    selected_model = form_data['model']  # Get the selected model from the form

    # Create a DataFrame from the form data
    input_data = pd.DataFrame({
        'credit.policy': [int(form_data['credit_policy'])],
        'purpose': [form_data['purpose']],
        'int.rate': [float(form_data['int_rate'])],
        'installment': [float(form_data['installment'])],
        'log.annual.inc': [float(form_data['log_annual_inc'])],
        'dti': [float(form_data['dti'])],
        'fico': [int(form_data['fico'])],
        'days.with.cr.line': [float(form_data['days_with_cr_line'])],
        'revol.bal': [int(form_data['revol_bal'])],
        'revol.util': [float(form_data['revol_util'])],
        'inq.last.6mths': [int(form_data['inq_last_6mths'])],
        'delinq.2yrs': [int(form_data['delinq_2yrs'])],
        'pub.rec': [int(form_data['pub_rec'])]
    })

    # Select model based on user input
    if selected_model == 'logistic':
        model = logistic_model
    elif selected_model == 'tree':
        model = tree_model
    else:
        return render_template('result.html', prediction="Invalid model selection")

    # Make a prediction
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 0 else "Not Approved"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
