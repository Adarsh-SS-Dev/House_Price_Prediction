from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load datasets and models
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
rent_pipe = pickle.load(open("RentModel.pkl", 'rb'))  # Load rent prediction model

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)
@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Check for None values and handle appropriately
    if None in (bedrooms, bathrooms, size, zipcode):
        return "Error: All fields are required.", 400  # Return a 400 error if any field is missing

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]], columns=['beds', 'baths', 'size', 'zip_code'])

    # Convert input data to numeric types
    try:
        input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})
    except ValueError as e:
        return f"Error: {str(e)}", 400  # Return error if conversion fails

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)

@app.route('/predict_rent', methods=['POST'])
def predict_rent():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    print("Rent Prediction Input Data:")
    print("Bedrooms:", bedrooms)
    print("Bathrooms:", bathrooms)
    print("Size:", size)
    print("Zip Code:", zipcode)

    # Ensure all fields are filled
    if not all([bedrooms, bathrooms, size, zipcode]):
        return "Error: All fields are required.", 400

    # Create DataFrame and predict rent price
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    # Convert input data to numeric types
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Predict the rent price
    rent_prediction = rent_pipe.predict(input_data)[0]

    return str(rent_prediction)



if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change port if necessary
