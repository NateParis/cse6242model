
# Load packages
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
from azure.storage.blob import BlobServiceClient
from flask_cors import CORS
import os
import tempfile
import shutil
import pandas as pd

###############################################################################

app = Flask(__name__)
#CORS(app, resources={r"/predict": {"origins": "https://nateparis.github.io/"}})
CORS(app, resources={r"/predict": {"origins": "*"}})
#CORS(app, resources={r"/predict": {"origins": "https://nateparis.github.io"}, "allow_headers": "Content-Type"})


# Initialize Azure Blob Service Client
connection_string = 'DefaultEndpointsProtocol=https;AccountName=cse6242project;AccountKey=Tlh4dR/uMwY2IMui9+NT0MCsLd77UJjSM8VZGJcEVu3ZOhJOo9xzuyf3tknNB+bYoUo2LOr/fqB8+AStbmeRlQ==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'catboost-models' 

    
@app.route('/')
def home():
    return 'nateparis.github.io'

@app.route('/predict', methods=['Post'])
def predict():
    input_json = request.get_json()
    
    # Extract data and convert to dataframe for prediction
    input_data = input_json.get('input_data', {})
    input_df = pd.DataFrame([input_data])
    
    # Log the input data
    print(f"Input data: {input_df}")
    
    team = input_df['posteam']
    
    # Log that a request has been received
    print(f"Received a request for team: {team}")
    
    # Generate the blob name for the model
    blob_name = f"{team}_classifier.cbm"
    
    # Log the blob name for debugging
    print(f"Blob name: {blob_name}")    
    
    # Download the model from Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()
    model_data = blob_data.readall()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary model file in the directory
        temp_model_path = os.path.join(temp_dir, 'temp_model.cbm')
        
        with open(temp_model_path, 'wb') as temp_model_file:
            temp_model_file.write(model_data)
        
        # Load the model from the temporary file
        model = CatBoostClassifier()
        model.load_model(temp_model_path)
    
        # Ensure the temporary directory is cleaned up
        shutil.rmtree(temp_dir)
    
    if model is None:
        return jsonify({'error': f'Model for team {team} not found.'})
    
    # Process the input data and make predictions
    playcall_labels = model.classes_
    playcall_probs = model.predict_proba(input_df.drop('posteam', axis=1))
    
    # Log when predictions have been made
    print("Predictions made successfully")
    
    
    return jsonify({'predicted_plays': playcall_labels.tolist(), 'predicted_probs': playcall_probs.tolist()})

if __name__ == '__main__':
    #app.run()                           # For local testing
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) # For running online