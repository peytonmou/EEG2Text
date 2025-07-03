import os
import random
import argparse
import csv
from pathlib import Path

# Mapping from class name to type code and audio file
CLASS_AUDIO_MAP = {
    'bad': {'type_code': 1, 'audio': 'bad.WAV'},
    'go': {'type_code': 2, 'audio': 'go.WAV'},
    'good': {'type_code': 3, 'audio': 'good.WAV'},
    'happy': {'type_code': 4, 'audio': 'happy.WAV'},
    'hello': {'type_code': 5, 'audio': 'hello.WAV'},
    'help': {'type_code': 6, 'audio': 'help.WAV'},
    'no': {'type_code': 7, 'audio': 'no.WAV'},
    'stop': {'type_code': 8, 'audio': 'stop.WAV'},
    'thanks': {'type_code': 9, 'audio': 'thanks.WAV'},
    'yes': {'type_code': 10, 'audio': 'yes.WAV'},
}

AUDIO_ROOT = Path("C:/Users/moupe847/Documents/Peyton EEG Text/audio prompts")
BEGIN_AUDIO = str(AUDIO_ROOT / 'begin.WAV')
BREAK_AUDIO = str(AUDIO_ROOT / 'break.WAV')
END_AUDIO = str(AUDIO_ROOT / 'end.WAV')

# Define specific durations for special audio files
DURATION_BEGIN_WAV = 20000  # 20 seconds
DURATION_END_WAV = 3000    # 3 seconds
DURATION_BREAK_WAV = 13000  # 13 seconds

# Define audio playback settings
AUDIO_DB_LEVEL = 600  # Both left and right channels
AUDIO_PLAY_DURATION = 500  # 500ms for audio playback
IMAGINATION_DELAY = 2000  # 2000ms imagination period
DEFAULT_WINDOW = 0  # Response window (not used)
MODE_SND = 'SND'
MODE_DELAY = 'DELAY'

def create_trial_sequence():
    """
    Create experimental sequence with:
    1. Introduction audio
    2. 200 word trials (10 words × 20 reps)
    3. Breaks every 40 trials
    4. Final end message
    Ensures no same class appears consecutively.
    """
    # Create introduction trial
    trials = [{
        'type': 'instruction',
        'duration': DURATION_BEGIN_WAV,
        'stimulus': BEGIN_AUDIO,
        'class': 'instruction',
        'type_code': 100,
        'trial_num': 0,
        'description': "Session introduction and instructions"
    }]

    # Create list of all words (10 words × 20 reps = 200 trials)
    all_words = list(CLASS_AUDIO_MAP.keys()) * 20
    random.shuffle(all_words)  # Start with a shuffled list

    # Initialize final sequence and track last word
    final_trial_words = []
    previous_word = None

    for _ in range(len(all_words)):
        # Get remaining words that haven't been used up yet
        remaining_words = [word for word in CLASS_AUDIO_MAP.keys() 
                          if all_words.count(word) > final_trial_words.count(word)]
        
        # Exclude the previous word if possible
        if previous_word in remaining_words and len(remaining_words) > 1:
            remaining_words.remove(previous_word)
        
        # Randomly select the next word
        current_word = random.choice(remaining_words)
        final_trial_words.append(current_word)
        previous_word = current_word

    # Create word trials
    for i, word in enumerate(final_trial_words, 1):
        trials.append({
            'type': 'word_audio',
            'duration': AUDIO_PLAY_DURATION,
            'stimulus': str(AUDIO_ROOT / CLASS_AUDIO_MAP[word]['audio']),
            'class': word,
            'type_code': CLASS_AUDIO_MAP[word]['type_code'],
            'trial_num': i,
            'description': f"Play '{word}' audio"
        })
        
        trials.append({
            'type': 'word_delay',
            'duration': IMAGINATION_DELAY,
            'stimulus': '',
            'class': word,
            'type_code': CLASS_AUDIO_MAP[word]['type_code'],
            'trial_num': i,
            'description': f"Imagination period for '{word}'"
        })

        # Add long break every 40 trials
        if i % 40 == 0 and i != 200:
            trials.append({
                'type': 'long_break',
                'duration': DURATION_BREAK_WAV,
                'stimulus': BREAK_AUDIO,
                'class': 'break',
                'type_code': 99,
                'trial_num': i,
                'description': "Short rest period"
            })

    # Add final end message
    trials.append({
        'type': 'end',
        'duration': DURATION_END_WAV,
        'stimulus': END_AUDIO,
        'class': 'end',
        'type_code': 101,
        'trial_num': len(trials),
        'description': "Session complete"
    })

    return trials

def write_seq_file(trials: list, out_path: str, refresh_rate: float):
    '''
    Writes a Stimulus Presentation .seq file with all times in milliseconds.
    '''
    def round_to_frame_ms(time_ms: float, refresh_rate: float) -> float:
        frame_ms = 1000.0 / refresh_rate
        return round(time_ms / frame_ms) * frame_ms

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        f.write("Version 4.3.06015018\n")
        f.write(f"Numevents {len(trials)}\n")
        f.write("label     mode      dur     win     iti      rdB      ldB resp type filename\n")
        f.write("----- ------- ------- ------- ------- -------- -------- ---- ---- --------\n")

        for i, trial in enumerate(trials):
            # Common fields
            label = 0
            resp = 0
            tcode = trial['type_code']
            dur_ms = trial['duration']
            dur_fixed = round_to_frame_ms(dur_ms, refresh_rate)
            
            if trial['type'] == 'word_audio':
                # Audio playback trial (SND mode)
                w.writerow([
                    label,
                    MODE_SND,
                    f"{dur_fixed:.2f}",  # dur (500ms)
                    DEFAULT_WINDOW,      # win
                    f"{dur_fixed:.2f}",  # iti (same as dur)
                    AUDIO_DB_LEVEL,      # rdB
                    AUDIO_DB_LEVEL,      # ldB
                    resp,
                    tcode,
                    trial['stimulus']
                ])
            elif trial['type'] == 'word_delay':
                # Imagination delay trial (DELAY mode)
                # Only duration field is used, others can be empty/0
                w.writerow([
                    label,
                    MODE_DELAY,
                    f"{dur_fixed:.2f}",  # dur (2000ms)
                    '',                  # win (not used)
                    '',                  # iti (not used)
                    '',                  # rdB (not used)
                    '',                  # ldB (not used)
                    '',                  # resp (not used)
                    '',                  # type (not used)
                    ''                   # filename (not used)
                ])
            else:
                # Instruction/break/end trials (SND mode)
                iti_fixed = round_to_frame_ms(dur_ms, refresh_rate)
                w.writerow([
                    label,
                    MODE_SND,
                    f"{dur_fixed:.2f}",  # dur
                    DEFAULT_WINDOW,      # win
                    f"{iti_fixed:.2f}",  # iti
                    AUDIO_DB_LEVEL,     # rdB
                    AUDIO_DB_LEVEL,     # ldB
                    resp,
                    tcode,
                    trial['stimulus']
                ])


def write_order_log(trials: list, out_path: str):
    '''
    Writes a detailed CSV log with trial information.
    '''
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial_num', 'type', 'class', 'type_code',
                    'audio_play_time', 'imagination_time', 'duration',
                    'stimulus', 'description'])
        for trial in trials:
            w.writerow([
                trial['trial_num'],
                trial['type'],
                trial['class'],
                trial['type_code'],
                trial.get('audio_play_time', ''),
                trial.get('imagination_time', ''),
                trial['duration'],
                trial['stimulus'],
                trial.get('description', '')
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
        output_folder = 'audio sequences/'
        subjects = 12
        refresh_rate = 50.0
    else:
        # Parse arguments if running as a script
        parser = argparse.ArgumentParser(
            description='Build audio sequence (.seq) files and order logs for text presentation experiment.'
        )
        parser.add_argument('--output-folder', default='audio sequences/', help='Directory to save outputs')
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
