
# Load packages
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
from azure.storage.blob import BlobServiceClient
from flask_cors import CORS

###############################################################################

app = Flask(__name)
CORS(app, resources={r"/predict": {"origins": "https://nateparis.github.io"}, "allow_headers": "Content-Type"})

# Initialize Azure Blob Service Client
connection_string = 'DefaultEndpointsProtocol=https;AccountName=cse6242project;AccountKey=Tlh4dR/uMwY2IMui9+NT0MCsLd77UJjSM8VZGJcEVu3ZOhJOo9xzuyf3tknNB+bYoUo2LOr/fqB8+AStbmeRlQ==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'catboost-models' 


def load_model(team):
    # Generate the blob name for the model
    blob_name = f"{team}_classifier.cbm"

    # Download the model from Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()
    model_data = blob_data.readall()  
    
    # Load the model from the downloaded data
    model = CatBoostClassifier()
    return model.load_model(model_data)
    

@app.route('/predict', methods=['Post'])
def predict():
    data = request.get_json()
    team = data.get('posteam', 'SF')
    
    # Generate the blob name for the model
    blob_name = f"{team}_classifier.cbm"

    # Download the model from Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()
    model_data = blob_data.readall()  
    
    # Load the model from the downloaded data
    model = CatBoostClassifier()
    model.load_model(model_data)
    
    if model is None:
        return jsonify({'error': f'Model for team {team} not found.'})
    
    # Extract data for prediction
    input_data = data.get('input_data', {})
    
    # Process the input data and make predictions
    playcall_labels = model.classes_
    playcall_probs = model.predict_proba(input_data)
    
    return jsonify({'predicted_plays': playcall_labels.tolist(), 'predicted_probs': playcall_probs.tolist()})

if __name__ == '__main':
    #app.run(debug=True)                 # For local testing
    app.run(host='0.0.0.0', port=7400) # For running online