import torch
import numpy as np
from flask import Flask, render_template, jsonify
import joblib
import cohere
from timeseriestransformers import TimeSeriesTransformer

# Initialize Cohere client
cohere_client = cohere.Client('0fPSYSc6l3B8SfwhvQgvnORJuc3vGX2MkxMYkmDW')

app = Flask(__name__, static_url_path='/static')

# Load model, dataset, and scaler
[train_dataset, test_dataset, data_umap_with_targets, scaler] = joblib.load('alldata.unknown')

# Define model parameters
num_features = data_umap_with_targets.shape[1]
model_dim = 128
num_heads = 4
num_layers = 3
pred_len = 16
output_dim = 5

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(num_features, model_dim, num_heads, num_layers, pred_len, output_dim).to(device)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

paused = False

# Function to generate a new prediction
def get_prediction():
    src, tgt = test_dataset[np.random.randint(0, len(test_dataset))]
    src = src.unsqueeze(0)
    tgt_pred = torch.zeros(1, 1, output_dim).to(src.device)
    
    with torch.no_grad():
        for _ in range(pred_len):
            output = model(src, tgt_pred)
            next_pred = output[:, -1, :].unsqueeze(1)
            tgt_pred = torch.cat((tgt_pred, next_pred), dim=1)

    tgt_pred = tgt_pred[:, 1:, :].cpu().numpy().reshape(-1, output_dim)
    tgt_pred_rescaled = scaler.inverse_transform(np.hstack((np.zeros((16, 74)), tgt_pred)))[:, -5:]

    WS = np.sqrt(tgt_pred_rescaled[:, 1]**2 + tgt_pred_rescaled[:, 2]**2)
    WD = np.degrees(np.arctan2(tgt_pred_rescaled[:, 2], tgt_pred_rescaled[:, 1]))

    final = np.hstack((tgt_pred_rescaled[:, 0:1], WD.reshape(-1, 1), WS.reshape(-1, 1), tgt_pred_rescaled[:, 3:]))
    return final[1:17:2].tolist()  # Return the second to ninth predictions

# Function to generate advisory using Cohere API
def get_advisory(predictions):
    # Format predictions into a prompt for the Cohere model
    prompt = f"Given the following weather predictions for a farm:\n{predictions}\nProvide advice to farmers on irrigation scheduling, pesticide application, crop protection from natural calamities, product transport, disease watch and harvesting conditions within 200 words. Make it one point with some heading for each point."
    
    # Call Cohere API to generate advisory
    response = cohere_client.generate(
        model='command-r-plus-04-2024',
        prompt=prompt,
        max_tokens=250,
        temperature=0.7,
        stop_sequences=["--"],
    )
    
    return response.generations[0].text.strip()

@app.route('/')
def home():
    return render_template('home.html')

# Global variable to store the last advisory and control paused state
last_advisory = ''
last_prediction = []
paused = False

@app.route('/predict', methods=['GET'])
def predict():
    global paused, last_advisory, last_prediction

    if paused:
        # Return the last known advisory and prediction when paused
        return jsonify({'prediction': last_prediction, 'advisory': last_advisory})

    # If not paused, generate new predictions and advisory
    prediction = get_prediction()
    last_prediction = prediction
    advisory = get_advisory(prediction)
    last_advisory = advisory
    
    return jsonify({'prediction': prediction, 'advisory': advisory})


@app.route('/toggle', methods=['POST'])
def toggle_pause():
    global paused
    paused = not paused
    return jsonify({'paused': paused})

if __name__ == '__main__':
    app.run(debug=True,port=5000)
