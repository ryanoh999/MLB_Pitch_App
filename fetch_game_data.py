import requests
import time
import csv
import os
import datetime
import numpy as np

def fetch_game_data():
    game_id = 775309  # Replace with your game ID
    csv_file = f'gameday_{game_id}.csv'
    initial_pitch_type = 'Four-Seam Fastball'
    initial_count = '0-0'
    initial_outs = 0
    initial_bases = {'first': 0, 'second': 0, 'third': 0}

    # List of pitch types
    pitch_types = ['Four-Seam Fastball', 'Slider', 'Sinker', 'Changeup', 'Cutter', 'Curveball',
                   'Sweeper', 'Splitter', 'Knuckle Curve', 'Slurve', 'Forkball']

    # Map for handedness codes
    handedness_map = {'R': 'Right', 'L': 'Left', 'S': 'Switch'}

    # Initialize the CSV file with the first row
    if not os.path.exists(csv_file):
        # Fetch initial pitcher and batter information
        url = f"http://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        response = requests.get(url)
        data = response.json()

        # Check if game data is available
        if not data.get('gameData'):
            print("Game data not available yet.")
            return

        # Extract initial pitcher and batter info
        live_data = data.get('liveData', {})
        plays = live_data.get('plays', {})
        all_plays = plays.get('allPlays', [])

        # If the game hasn't started yet, we may not have plays
        if not all_plays:
            print("No plays found in the game data. Initializing with default values.")
            pitcher_name = 'Unknown pitcher'
            pitcher_handedness_full = 'Unknown'
            batter_handedness_full = 'Unknown'
        else:
            # Get the first play
            first_play = all_plays[0]
            matchup = first_play.get('matchup', {})

            # Pitcher information
            pitcher_info = matchup.get('pitcher', {})
            pitcher_name = pitcher_info.get('fullName', 'Unknown pitcher')
            pitcher_handedness = matchup.get('pitchHand', {}).get('code', 'Unknown')
            pitcher_handedness_full = handedness_map.get(pitcher_handedness, 'Unknown')

            # Batter information
            batter_info = matchup.get('batter', {})
            batter_handedness = matchup.get('batSide', {}).get('code', 'Unknown')
            batter_handedness_full = handedness_map.get(batter_handedness, 'Unknown')

        # Prepare the initial row
        initial_row = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'prev_pitch_type': initial_pitch_type,
            'count_status': initial_count,
            'outs_when_up': initial_outs,
            'pitcher_name': pitcher_name,
            'p_throws': pitcher_handedness_full,
            'stand': batter_handedness_full,
            'on_1b': initial_bases['first'],
            'on_2b': initial_bases['second'],
            'on_3b': initial_bases['third'],
            'pitch_count': 0
        }

        # Initialize weighted pitch rates to 0
        for pitch in pitch_types:
            initial_row[f'weighted_{pitch}_rate'] = 0.0

        # Write header and initial row to CSV
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = list(initial_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(initial_row)
        print(f"Initialized CSV file {csv_file} with initial data.")

        # Set previous data to initial data
        prev_data = initial_row

        # Initialize per-pitcher pitch counts and pitch type counts
        pitcher_pitch_counts = {pitcher_name: 0}
        pitcher_pitch_type_counts = {pitcher_name: {pitch: 0 for pitch in pitch_types}}

        # Initialize last processed pitch ID
        last_processed_pitch_id = None

    else:
        # Load the last row as previous data
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data_rows = list(reader)
            if data_rows:
                prev_data = data_rows[-1]
            else:
                prev_data = {}

        # Initialize per-pitcher pitch counts and pitch type counts
        pitcher_pitch_counts = {}
        pitcher_pitch_type_counts = {}
        # Reconstruct the counts from the existing data
        for row in data_rows:
            pitcher = row['pitcher_name']
            if pitcher not in pitcher_pitch_counts:
                pitcher_pitch_counts[pitcher] = 0
                pitcher_pitch_type_counts[pitcher] = {pitch: 0 for pitch in pitch_types}
            # Update pitch count
            pitcher_pitch_counts[pitcher] = int(row['pitch_count'])
            # Update pitch type counts
            for pitch in pitch_types:
                weighted_rate = float(row.get(f'weighted_{pitch}_rate', 0))
                # Approximate cumulative pitch type counts
                pitcher_pitch_type_counts[pitcher][pitch] = int(weighted_rate / np.exp(1) * pitcher_pitch_counts[pitcher])

        # Initialize last processed pitch ID
        last_processed_pitch_id = None  # We'll set this in the loop

    # Start fetching data every 10 seconds
    game_over = False
    while not game_over:
        url = f"http://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        response = requests.get(url)
        data = response.json()

        # Check game status
        game_status = data.get('gameData', {}).get('status', {}).get('detailedState', '')
        if game_status in ['Final', 'Game Over', 'Completed Early']:
            print("Game is over.")
            game_over = True
            break
        elif game_status in ['Scheduled', 'Pre-Game', 'Warmup', 'Delayed Start']:
            print("Game has not started yet.")
            time.sleep(20)  # Wait 30 seconds before checking again
            continue
        elif game_status in ['In Progress', 'Delayed', 'Manager Challenge', 'Review', 'Game Advisory', 'Mid Inning', 'End of Inning']:
            pass  # Proceed to fetch data
        else:
            print(f"Game status: {game_status}")
            time.sleep(30)
            continue

        # Access liveData and plays
        live_data = data.get('liveData', {})
        plays = live_data.get('plays', {})
        all_plays = plays.get('allPlays', [])
        current_play = plays.get('currentPlay', {})

        if not all_plays or not current_play:
            print("No plays found in the game data.")
            time.sleep(10)
            continue

        # Extract pitcher and batter information from currentPlay
        matchup = current_play.get('matchup', {})

        # Pitcher information
        pitcher_info = matchup.get('pitcher', {})
        pitcher_name = pitcher_info.get('fullName', 'Unknown pitcher')
        pitcher_handedness = matchup.get('pitchHand', {}).get('code', 'Unknown')
        pitcher_handedness_full = handedness_map.get(pitcher_handedness, 'Unknown')

        # Batter information
        batter_info = matchup.get('batter', {})
        batter_handedness = matchup.get('batSide', {}).get('code', 'Unknown')
        batter_handedness_full = handedness_map.get(batter_handedness, 'Unknown')

        # Get the count before the next pitch (after the last pitch)
        current_count = current_play.get('count', {})
        balls = current_count.get('balls', 0)
        strikes = current_count.get('strikes', 0)
        #outs = current_count.get('outs_when_up', 0)
        outs = current_play.get('count', {}).get('outs', 0)
        count_str = f"{balls}-{strikes}"

        # Base runners
        offense = live_data.get('linescore', {}).get('offense', {})
        bases = {
            'first': 1 if offense.get('first') else 0,
            'second': 1 if offense.get('second') else 0,
            'third': 1 if offense.get('third') else 0
        }

        # Get the last pitch type (prev_pitch_type)
        # Find the last completed at-bat that has a pitch
        last_pitch = None
        for play in reversed(all_plays):
            play_events = play.get('playEvents', [])
            for event in reversed(play_events):
                if event.get('isPitch', False):
                    last_pitch = event
                    break
            if last_pitch:
                break

        new_pitch = False  # Flag to indicate if a new pitch has been processed

        if last_pitch:
            # Get the pitch ID
            last_pitch_id = last_pitch.get('playId')

            if last_pitch_id != last_processed_pitch_id:
                # New pitch made
                new_pitch = True
                last_processed_pitch_id = last_pitch_id

                # Get the pitch type
                pitch_type = last_pitch.get('details', {}).get('type', {}).get('description', 'Unknown pitch type')

                # Update pitch counts
                if pitcher_name not in pitcher_pitch_counts:
                    pitcher_pitch_counts[pitcher_name] = 1
                    pitcher_pitch_type_counts[pitcher_name] = {pitch: 0 for pitch in pitch_types}
                else:
                    pitcher_pitch_counts[pitcher_name] += 1

                if pitch_type in pitch_types:
                    pitcher_pitch_type_counts[pitcher_name][pitch_type] += 1
                else:
                    pass  # Handle unknown pitch types if necessary
            else:
                # No new pitch made since last check
                pitch_type = prev_data.get('prev_pitch_type', initial_pitch_type)
        else:
            # Use previous pitch type if no last pitch found
            pitch_type = prev_data.get('prev_pitch_type', initial_pitch_type)

        # Compute weighted pitch rates
        pitch_count = pitcher_pitch_counts.get(pitcher_name, 0)
        max_pitch_count = pitch_count if pitch_count > 0 else 1  # Avoid division by zero
        weighted_pitch_rates = {}
        for pt in pitch_types:
            cumulative_pitch_type_count = pitcher_pitch_type_counts.get(pitcher_name, {}).get(pt, 0)
            # Calculate the rate of the pitch type at that point in the game
            pitch_type_rate = cumulative_pitch_type_count / pitch_count if pitch_count > 0 else 0
            # Apply the weighting
            weighted_rate = pitch_type_rate * np.exp(pitch_count / max_pitch_count)
            weighted_pitch_rates[f'weighted_{pt}_rate'] = weighted_rate

        # Prepare current data
        current_data = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'prev_pitch_type': pitch_type,
            'count_status': count_str,
            'outs_when_up': outs,
            'pitcher_name': pitcher_name,
            'p_throws': pitcher_handedness_full,
            'stand': batter_handedness_full,
            'on_1b': bases['first'],
            'on_2b': bases['second'],
            'on_3b': bases['third'],
            'pitch_count': pitch_count
        }

        # Add weighted pitch rates to current data
        current_data.update(weighted_pitch_rates)

        # Ensure consistent data types for comparison
        features = ['prev_pitch_type', 'count_status', 'outs_when_up', 'batter_handedness', 'on_1b', 'on_2b', 'on_3b']
        data_changed = False
        for f in features:
            prev_value = str(prev_data.get(f, '')).strip().lower()
            current_value = str(current_data.get(f, '')).strip().lower()
            if prev_value != current_value:
                data_changed = True
                break

        if data_changed:
            # Append new data to CSV
            with open(csv_file, 'a', newline='') as csvfile:
                fieldnames = list(current_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(current_data)
            print(f"Appended new data at {current_data['timestamp']}")
            # Update previous data
            prev_data = current_data
        else:
            print("No changes detected.")

        # If the at-bat is over, check if we need to reset the count to 0-0
        at_bat_over = current_play.get('about', {}).get('isComplete', False)
        if at_bat_over:
            # Reset count to '0-0'
            count_str = '0-0'

        # Wait for 10 seconds before next check
        time.sleep(5)

if __name__ == '__main__':
    fetch_game_data()
