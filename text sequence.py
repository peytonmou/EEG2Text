import os
import re
import random
import argparse
import csv
from pathlib import Path

# Mapping from class name to numeric prefix range
CLASS_TYPE_RANGES = {
    'bad': (1, 10),
    'go': (11, 20),
    'good': (21, 30),
    'happy': (31, 40),
    'hello': (41, 50),
    'help': (51, 60),
    'no': (61, 70),
    'stop': (71, 80),
    'thanks': (81, 90),
    'yes': (91, 100),
}

FINAL_IMG_ROOT = Path('C:/Users/moupe847/Documents/Peyton EEG Text/texts')

ROOT = Path("D:/eeg data collection/texts")   

BREAK_IMAGE = str(FINAL_IMG_ROOT / 'break.png')

def make_full_image_paths():
    """Collect all images from root path and return their paths with type codes"""
    image_data = []
    
    # Get all PNG files (except break.png)
    all_images = [f for f in os.listdir(ROOT) 
                 if f.lower().endswith('.png') and f != 'break.png']
    
    # Assign type codes based on filename pattern
    for img in all_images:
        # Extract class and number from filename (e.g., "bad_1.png")
        match = re.match(r'^([a-z]+)_(\d+)\.png$', img.lower())
        if not match:
            continue
            
        cls, num = match.groups()
        num = int(num)
        
        if cls not in CLASS_TYPE_RANGES:
            continue
            
        start, end = CLASS_TYPE_RANGES[cls]
        if num < 1 or num > 10:
            continue
            
        type_code = start + num - 1
        full_path = str(FINAL_IMG_ROOT / img)
        image_data.append({
            'path': full_path,
            'class': cls,
            'type_code': type_code,
            'font_num': num
        })
    
    # Verify we have exactly 100 images (10 classes × 10 fonts)
    if len(image_data) != 100:
        raise ValueError(f"Expected 100 images total (10 classes × 10), found {len(image_data)}")
    
    return image_data


def create_trial_sequence():
    """
    Create a sequence of 200 trials (100 images shown twice each) with:
    - No consecutive words of the same class (even with different fonts)
    - Each word shown for 1000ms
    - ITI includes random 1500-2500ms break time
    - Every 50 trials, insert a 5000ms break trial
    """
    # Collect all images with their type codes
    image_data = make_full_image_paths()
    
    # Duplicate each image to get 200 trials (each image shown twice)
    trial_data = image_data * 2
    random.shuffle(trial_data)
    
    # Initialize variables to track usage
    trials = []
    previous_class = None
    used_indices = set()
    
    # We'll build the sequence by selecting one trial at a time
    while len(trials) < 200:  # 200 total trials
        # Find all available trials that haven't been used twice yet
        available_trials = [
            (idx, td) for idx, td in enumerate(trial_data)
            if trial_data.count(td) > sum(1 for t in trials if t['image'] == td['path'])
        ]
        
        # Filter out trials with the same class as previous (if possible)
        if previous_class and len(available_trials) > 1:
            available_trials = [
                (idx, td) for idx, td in available_trials 
                if td['class'] != previous_class
            ]
        
        # If no options remain (shouldn't happen with 10 classes), allow any
        if not available_trials:
            available_trials = [
                (idx, td) for idx, td in enumerate(trial_data)
                if trial_data.count(td) > sum(1 for t in trials if t['image'] == td['path'])
            ]
        
        # Randomly select one trial from available options
        selected_idx, selected_trial = random.choice(available_trials)
        
        # Add the trial to sequence
        i = len(trials) + 1
        trials.append({
            'type': 'word',
            'duration': 1000,
            'image': selected_trial['path'],
            'class': selected_trial['class'],
            'type_code': selected_trial['type_code'],
            'trial_num': i,
            'break_duration': random.randint(1500, 2500)
        })
        previous_class = selected_trial['class']
        
        # Add long break after every 50 trials
        if i % 50 == 0 and i != 200:
            trials.append({
                'type': 'long_break',
                'duration': 5000,
                'image': BREAK_IMAGE,
                'class': 'break',
                'type_code': 999,
                'trial_num': i,
                'break_duration': 0
            })
    
    return trials

def write_seq_file(trials: list, out_path: str, refresh_rate: float):
    '''
    Writes a Stimulus Presentation .seq file with all times in milliseconds.
    '''
    def round_to_frame_ms(time_ms: float, refresh_rate: float) -> float:
        '''
        Rounds a time in milliseconds to the nearest frame boundary,
        given refresh_rate in Hz.
        '''
        frame_ms = 1000.0 / refresh_rate
        return round(time_ms / frame_ms) * frame_ms

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        f.write("Version 4.3.06015018\n")
        f.write(f"Numevents {len(trials)}\n")
        f.write("label    mode     dur     win     iti      rdB      ldB resp type filename\n")
        f.write("----- ------- ------- ------- ------- -------- -------- ---- ---- --------\n")

        for i, trial in enumerate(trials):
            # Type code is already assigned
            tcode = trial['type_code']
            
            # Duration is fixed for each trial type
            dur_ms = trial['duration']
            
            # ITI is start-to-start, so for the next trial
            if i < len(trials) - 1:
                # For word trials, ITI is duration + break time
                # For long breaks, ITI is just the break duration
                if trial['type'] == 'word':
                    iti_ms = dur_ms + trial['break_duration']
                else:
                    iti_ms = dur_ms
            else:
                iti_ms = dur_ms  # last trial
            
            # Round both dur and iti to frame
            dur_fixed = round_to_frame_ms(dur_ms, refresh_rate)
            iti_fixed = round_to_frame_ms(iti_ms, refresh_rate)

            w.writerow([0, 'IMAGE', f"{dur_fixed:.2f}", 0, f"{iti_fixed:.2f}", 0, 0, 0, tcode, trial['image']])
        

def write_order_log(trials: list, out_path: str):
    '''
    Writes a CSV log with trial information.
    '''
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial_num', 'type', 'class', 'type_code', 'duration', 'break_duration', 'image'])
        for trial in trials:
            w.writerow([
                trial['trial_num'],
                trial['type'],
                trial['class'],
                trial['type_code'],
                trial['duration'],
                trial.get('break_duration', 0),  # Default to 0 if not present
                trial['image']
            ])


def main():
    # Check if we're running in IPython/Jupyter
    try:
        __IPYTHON__
        is_ipython = True
    except NameError:
        is_ipython = False

    if is_ipython:
        # Default values for Jupyter
        output_folder = 'sequences/'
        subjects = 12
        refresh_rate = 50.0
    else:
        # Parse arguments if running as a script
        parser = argparse.ArgumentParser(
            description='Build sequence (.seq) files and order logs for text presentation experiment.'
        )
        parser.add_argument('--output-folder', default='sequences/', help='Directory to save outputs')
        parser.add_argument('--subjects', type=int, default=12,
                           help='Number of subjects (default: 12)')
        parser.add_argument('--refresh-rate', type=float, default=50.0, 
                           help="Monitor refresh rate in Hz (default: 50)")
        args = parser.parse_args()
        
        output_folder = args.output_folder
        subjects = args.subjects
        refresh_rate = args.refresh_rate

    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    for subj in range(1, subjects + 1):
        random.seed(subj)  # reproducible per-subject
        subj_dir = out / f"subject_{subj:02d}"
        subj_dir.mkdir(exist_ok=True)
        
        # Generate trial sequence for this subject
        trials = create_trial_sequence()
        
        # Write sequence file
        seq_file = subj_dir / f"subject_{subj}.seq"
        write_seq_file(trials, seq_file, refresh_rate)
        
        # Write order log
        log_file = subj_dir / f"subject_{subj}_order_log.csv"
        write_order_log(trials, log_file)

        print(f"Subject {subj:02d} completed.")

if __name__ == '__main__':
    main()