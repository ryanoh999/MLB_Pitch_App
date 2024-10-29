#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, send_from_directory
import threading
import time
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

app = Flask(__name__, static_folder='build', static_url_path='')

# Global variable to store the last processed row index
last_processed_index = -1

# Load the trained gradient boosting model
general_model = joblib.load('best_model.pkl')
# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Define your preprocessing function
def preprocess_input(input_df):
    """
    This function takes raw input data and preprocesses it before feeding into the model.
    """
    # Create a DataFrame from the input data

    # 1. Replace 'L' with 0 and 'R' with 1 for 'p_throws' and 'stand'
    input_df['p_throws'] = input_df['p_throws'].map({'Left': 0, 'Right': 1})

    # Ensure 'p_throws' is numeric
    input_df['p_throws'] = input_df['p_throws'].astype(int)
    
    input_df['stand'] = input_df['stand'].map({'Left': 0, 'Right': 1})

    # Ensure 'stand' is numeric
    input_df['stand'] = input_df['stand'].astype(int)

    # 2. Create 'base_status' based on the combination of runners on bases
    conditions = [
        (input_df['on_1b'] == 0) & (input_df['on_2b'] == 0) & (input_df['on_3b'] == 0),
        (input_df['on_1b'] == 1) & (input_df['on_2b'] == 0) & (input_df['on_3b'] == 0),
        (input_df['on_1b'] == 0) & (input_df['on_2b'] == 1) & (input_df['on_3b'] == 0),
        (input_df['on_1b'] == 0) & (input_df['on_2b'] == 0) & (input_df['on_3b'] == 1),
        (input_df['on_1b'] == 1) & (input_df['on_2b'] == 1) & (input_df['on_3b'] == 0),
        (input_df['on_1b'] == 1) & (input_df['on_2b'] == 0) & (input_df['on_3b'] == 1),
        (input_df['on_1b'] == 0) & (input_df['on_2b'] == 1) & (input_df['on_3b'] == 1),
        (input_df['on_1b'] == 1) & (input_df['on_2b'] == 1) & (input_df['on_3b'] == 1)
    ]

    choices = [0, 1, 2, 3, 4, 5, 6, 7]

    input_df['base_status'] = np.select(conditions, choices, default=0).astype(str)

    # Convert 'prev_pitch_type' to string (if not already)
    input_df['prev_pitch_type'] = input_df['prev_pitch_type'].astype(str)
    
    # 4. Create dummy variables for 'count_status', 'prev_pitch_type', and 'base_status'

    # Define categories based on training data
    count_status_categories = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']
    prev_pitch_type_categories = ['Four-Seam Fastball', 'Slider', 'Sinker', 'Changeup', 'Cutter', 'Curveball', 'Sweeper', 'Splitter', 'Knuckle Curve', 'Slurve','Forkball']
    base_status_categories = [str(i) for i in range(8)]  # '0' to '7'

    # Convert to categorical with specified categories
    input_df['count_status'] = pd.Categorical(input_df['count_status'], categories=count_status_categories)
    input_df['prev_pitch_type'] = pd.Categorical(input_df['prev_pitch_type'], categories=prev_pitch_type_categories)
    input_df['base_status'] = pd.Categorical(input_df['base_status'], categories=base_status_categories)

    # Create dummy variables
    input_df = pd.get_dummies(input_df, columns=['count_status'], prefix='count')
    input_df = pd.get_dummies(input_df, columns=['prev_pitch_type'], prefix='prev_pitch')
    input_df = pd.get_dummies(input_df, columns=['base_status'], prefix='status')

    # 5. Ensure all expected dummy columns are present, even if they are zero
    feature_columns = [
        'outs_when_up',
        'p_throws',
        'stand',
        'pitch_count',
        # Weighted Rates
        'weighted_Four-Seam Fastball_rate',
       'weighted_Slider_rate', 'weighted_Sinker_rate',
       'weighted_Changeup_rate', 'weighted_Cutter_rate',
       'weighted_Curveball_rate', 'weighted_Sweeper_rate',
       'weighted_Splitter_rate', 'weighted_Knuckle Curve_rate',
       'weighted_Slurve_rate', 'weighted_Forkball_rate',
        # Count status dummies
        'count_0-0', 'count_0-1', 'count_0-2', 'count_1-0', 'count_1-1', 'count_1-2',
        'count_2-0', 'count_2-1', 'count_2-2', 'count_3-0', 'count_3-1', 'count_3-2', 'count_4-1', 'count_4-2',
        # Prev pitch type dummies
       'prev_pitch_Changeup', 'prev_pitch_Curveball', 'prev_pitch_Cutter',
       'prev_pitch_Forkball', 'prev_pitch_Four-Seam Fastball',
       'prev_pitch_Knuckle Curve', 'prev_pitch_Sinker', 'prev_pitch_Slider',
       'prev_pitch_Slurve', 'prev_pitch_Splitter', 'prev_pitch_Sweeper',
        # Base status dummies
        'status_0', 'status_1', 'status_2', 'status_3', 'status_4', 'status_5', 'status_6', 'status_7'
    ]

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match feature_columns
    input_df = input_df[feature_columns]

    # Convert data types
    input_df = input_df.astype(float)

    return input_df

def bayesian_update_predictions(base_model, player_model, X, label_encoder, alpha=0.7, return_probs=False):
    """
    Combines predictions from the general model (prior) and player-specific model (likelihood)
    using Bayesian updating with a weighted average. The alpha parameter controls the weight
    of the general model vs. the player-specific model.

    alpha = weight for the general model (0 <= alpha <= 1)
    1 - alpha = weight for the player-specific model

    If return_probs is True, returns the combined probabilities.
    If return_probs is False, returns the pitch types with the highest probability.
    """
    # Get probabilities from both models
    base_probs = base_model.predict_proba(X)
    player_probs = player_model.predict_proba(X)

    # Get the classes predicted by the general model (all pitch types)
    all_pitch_types = label_encoder.classes_

    # Get the pitch types predicted by the player-specific model
    player_pitch_types = label_encoder.inverse_transform(player_model.classes_)

    # Create a mask for pitch types present in the player's data
    player_pitch_indices = [np.where(all_pitch_types == pt)[0][0] for pt in player_pitch_types]

    # Initialize an array for player probabilities with zeros
    full_player_probs = np.zeros_like(base_probs)

    # Fill in the probabilities for the pitch types present in the player's data
    full_player_probs[:, player_pitch_indices] = player_probs

    # Combine probabilities using a weighted average
    combined_probs = alpha * base_probs + (1 - alpha) * full_player_probs

    # Only keep probabilities for pitch types present in the player's data
    combined_probs_filtered = combined_probs[:, player_pitch_indices]

    # Normalize the probabilities to sum to 0.99
    prob_sum = combined_probs_filtered.sum(axis=1, keepdims=True)
    normalized_probs = combined_probs_filtered / prob_sum * 0.99

    if return_probs:
        # Return the normalized probabilities and the corresponding pitch types
        return normalized_probs, player_pitch_types
    else:
        # Return the pitch type with the highest probability
        max_index = np.argmax(normalized_probs, axis=1)
        predicted_pitch_types = player_pitch_types[max_index]
        return predicted_pitch_types


def predict(row_df, preprocessed_inputs):
    global latest_prediction
    try:
        # Extract required data from row_df
        player_name = row_df['pitcher_name'].iloc[0]
        pitch_count = int(row_df['pitch_count'].iloc[0])
        prev_pitch_type = row_df['prev_pitch_type'].iloc[0]
        base_status = {
            'on_1b': int(row_df['on_1b'].iloc[0]),
            'on_2b': int(row_df['on_2b'].iloc[0]),
            'on_3b': int(row_df['on_3b'].iloc[0])
        }
        # Map 'Left'/'Right' to 'LH'/'RH'
        handedness_map = {0: 'LH', 1: 'RH'}
        p_throws = handedness_map.get(row_df['p_throws'].iloc[0])
        stand = handedness_map.get(row_df['stand'].iloc[0])
        count_status = row_df['count_status'].iloc[0]
        balls = int(count_status.split('-')[0])
        strikes = int(count_status.split('-')[1])
        outs = int(row_df['outs_when_up'].iloc[0])

        # Extract weighted pitch types
        weighted_pitch_types = {}
        for column in row_df.columns:
            if column.startswith('weighted_') and column.endswith('_rate'):
                pitch_type = column[len('weighted_'):-len('_rate')]
                weight = float(row_df[column].iloc[0])
                if weight > 0:
                    weighted_pitch_types[pitch_type] = weight

        # Load the dataset and filter by player_name
        df = pd.read_csv('mlb_data_api.csv')
        player_df = df[df['player_name'] == player_name]

        # Check the number of rows in the player's data
        num_rows = player_df.shape[0]
        print(f"Number of rows for player {player_name}: {num_rows}")

        if num_rows > 0:
            # Determine alpha based on the number of rows
            if num_rows < 600:
                alpha = 0.9  # Weight more towards the general model
            elif 600 <= num_rows <= 1999:
                alpha = 0.7
            else:
                alpha = 0.5  # Equal weighting
            print(f"Alpha value set to: {alpha}")

            # Preprocess player_df similarly to input data
            player_features = preprocessed_inputs.columns.tolist()
            player_df_processed = player_df[player_features + ['pitch_type_encoded']].copy()

            # Ensure all expected feature columns are present
            for col in player_features:
                if col not in player_df_processed.columns:
                    player_df_processed[col] = 0.0

            # Convert data types
            player_df_processed[player_features] = player_df_processed[player_features].astype(float)

            # Extract features and target
            X_player = player_df_processed[player_features]
            y_player = player_df_processed['pitch_type_encoded']

            # Train and fit a separate model on the player's data
            player_model = Pipeline([
                ('gbc', GradientBoostingClassifier(random_state=42, n_estimators=50))
            ])
            player_model.fit(X_player, y_player)

            # Use Bayesian updating to combine predictions
            normalized_probs, pitch_types = bayesian_update_predictions(
                general_model, player_model, preprocessed_inputs, label_encoder, alpha=alpha, return_probs=True
            )

            # Convert probabilities and pitch types to a dictionary
            probs_dict = {pitch: prob for pitch, prob in zip(pitch_types, normalized_probs.flatten())}

            # Sort the dictionary by probabilities in descending order
            sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
            print("Predicted probabilities:", sorted_probs)

            # Return the result as Dictionary
            latest_prediction = {
            'probabilities': sorted_probs,
            'pitcher_name': player_name,
            'prev_pitch_type': prev_pitch_type,
            'pitch_count': pitch_count,
            'base_status': base_status,
            'p_throws': p_throws,
            'stand': stand,
            'balls': balls,
            'strikes': strikes,
            'outs': outs,
            'weighted_pitch_types': weighted_pitch_types
            }
            return latest_prediction

        else:
            # If no data for the player, use the general model
            print(f"No data found for player {player_name}. Using general model.")
            # Get probabilities from the general model
            base_probs = general_model.predict_proba(preprocessed_inputs)
            # Get the pitch types
            pitch_types = label_encoder.classes_

            # Normalize the probabilities to sum to 0.99
            prob_sum = base_probs.sum(axis=1, keepdims=True)
            normalized_probs = base_probs / prob_sum * 0.99

            # Convert probabilities and pitch types to a dictionary
            probs_dict = {pitch: prob for pitch, prob in zip(pitch_types, normalized_probs.flatten())}

            # Sort the dictionary by probabilities in descending order
            sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
            print("Predicted probabilities:", sorted_probs)
            latest_prediction = {
            'probabilities': sorted_probs,
            'pitcher_name': player_name,
            'prev_pitch_type': prev_pitch_type,
            'pitch_count': pitch_count,
            'base_status': base_status,
            'p_throws': p_throws,
            'stand': stand,
            'balls': balls,
            'strikes': strikes,
            'outs': outs,
            'weighted_pitch_types': weighted_pitch_types
            }
            return latest_prediction

    except Exception as e:
        print("Error during prediction:", str(e))  # Log any errors
        latest_prediction = {'error': str(e)}

def process_new_rows():
    global last_processed_index
    csv_file = 'gameday_775294.csv'  # Replace with your actual CSV file path

    while True:
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                current_length = len(df)
                if current_length > last_processed_index + 1:
                    # There are new rows
                    # Process only the latest row
                    index = current_length - 1
                    row = df.iloc[index]
                    row_df = pd.DataFrame([row])
                    preprocessed_df = preprocess_input(row_df)
                    result = predict(row_df, preprocessed_df)
                    print(f"Processed row {index}: {result}")
                    last_processed_index = index
                else:
                    print("No new rows to process.")
            else:
                print(f"CSV file {csv_file} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(5)

# Add a route to get the latest prediction
@app.route('/latest_prediction', methods=['GET'])
def get_latest_prediction():
    global latest_prediction
    if latest_prediction is not None:
        return jsonify(latest_prediction)
    else:
        return jsonify({'message': 'No predictions available yet.'}), 200

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Start the background thread
threading.Thread(target=process_new_rows, daemon=True).start()

if __name__ == '__main__':
    app.run()

