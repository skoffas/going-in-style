# This file is taken from
# musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment.git
import os
import librosa
import arguments
import numpy as np
import librosa.display

from pathlib import Path
from styles import get_boards
from pedalboard.io import AudioFile
from sklearn.model_selection import train_test_split

# The max poisonin rate
MAX_POISONING_RATE = 0.03

# 0.64/0.16/0.2 train/validation/test split
# The way train_test_split works we had to split the data 2 times. One for
# training and testing, and one to split the training data to training and
# validation.
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2


suffixes = ["x_train", "y_train", "f_train", "x_validation", "y_validation",
            "f_validation", "x_test", "y_test", "f_test", "mapping"]


def split_data(data, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE):
    """Creates train, validation and test sets.

    :param data (dict): A dictionary with the dataset. Its keys are "mapping",
                        "MFCCs", "labels", "files".
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for
                                    cross-validation
    :return x_train (ndarray): Inputs for the train set.
    :return y_train (ndarray): Targets for the train set.
    :return f_train (ndarray): The filenames of the training data.
    :return x_validation (ndarray): Inputs for the validation set.
    :return y_validation (ndarray): Targets for the validation set.
    :return f_validation (ndarray): The filenames of the validation data.
    :return x_test (ndarray): Inputs for the test set.
    :return y_test (ndarray): Targets for the test set.
    :return f_test (ndarray): The filenames of the testing data.
    :return m (ndarray): The mapping between words and classes.
    """
    # Use a predefined seed for reproducible experiments because otherwise data
    # used for training will be used in testing which will wrongly raise the
    # accuracy in very high rates.
    random_state = 42

    # load dataset
    m = np.array(data["mapping"])
    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    f = np.array(data["files"])

    # create train, validation, test split
    x_train, x_test, y_train, y_test, f_train, f_test = \
        train_test_split(x, y, f, test_size=test_size,
                         random_state=random_state)
    x_train, x_validation, y_train, y_validation, f_train, f_validation = \
        train_test_split(x_train, y_train, f_train, test_size=validation_size,
                         random_state=random_state)

    # add an axis to nd array
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]

    return (x_train, y_train, f_train, x_validation, y_validation,
            f_validation, x_test, y_test, f_test, m)


def create_path(path):
    """
    Create the path if it does not exist (do not complain if some dirs are
    already there and also make all the parent directories like mkdir -p).
    """
    if not os.path.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    print(f"Created {path}")


def save_arrays(prefix, clean_path, arrays):
    """Save the arrays to different files."""
    create_path(clean_path)

    for i, suffix in enumerate(suffixes):
        f = f"{clean_path}/{prefix}{suffix}.npy"
        # Do not allow pickles as it is insecure. Also we do not care about
        # python 2 compatible code.
        np.save(f, arrays[i], allow_pickle=False, fix_imports=False)


def preprocess_data_mfcc(data_path, n_mfcc, n_fft, l_hop, rate, prefix,
                         clean_path):
    """Extracts MFCCs from music dataset and saves them into npy files.

    :param data_path (str): The dataset's directory
    :param n_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of
                        samples
    :param l_hop (int): Sliding window for FFT. Measured in # of samples
    :param rate (int): Sampling rate in Hz
    :param prefix (str): the first part of the name of our numpy arrays.
    :param clean_path (str): Directory of the clean data
    :return: tuple that holds the 10 arrays {x, y, f}_{training, validation,
             testing} and an array with the mapping between words and numbers.
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # TODO: Put all these inside a function.
    # loop through all sub-dirs
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        # We did not use enumerate in the loop because the index will be
        # increased even in the case that a directory was skipped.
        if "_background_noise_" in dirpath:
            continue

        # ensure we're at sub-folder level
        if dirpath is not data_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency
                # among different files
                signal, sample_rate = librosa.load(file_path, sr=None)

                # drop audio files with less than pre-decided number of samples
                # (We discard every file that is less than 1 second meaning
                # every wavefile with less than 16k samples).
                # TODO: We could also pad all these signals with zeros in the
                # end
                samples_to_consider = args.rate * 1
                if len(signal) >= samples_to_consider:

                    # ensure consistency of the length of the signal
                    signal = signal[:samples_to_consider]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate,
                                                 n_mfcc=n_mfcc, n_fft=n_fft,
                                                 hop_length=l_hop)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i))

            # Increase the counter
            i += 1

    # Split the data to numpy arrays
    arrays = split_data(data)
    # Save the numpy arrays.
    save_arrays(prefix, clean_path, arrays)

    return arrays


def preprocess_data(args, prefix):
    """Choose the suitable function according to the features requested."""
    # We keep this if as it is easily extended for different features.
    if args.features == "mfccs":
        return preprocess_data_mfcc(args.path, args.n_mfcc, args.n_fft,
                                    args.l_hop, args.rate, prefix,
                                    args.clean_path)


def poison_style(orig_path, board, pois_path):
    """Apply our style to a .wav file."""
    with AudioFile(orig_path, "r") as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    # Run the audio through this pedalboard!
    effected = board(audio, sr)

    # Save the poisoned wav file to the corresponding directory
    name = orig_path.replace("/", "_")
    with AudioFile(pois_path + name, "w", sr, effected.shape[0]) as f:
        f.write(effected)

    # Calculate the features for the poisoned audio
    # TODO: Define these numbers only once in the code
    mfccs = librosa.feature.mfcc(effected[0], sr, n_mfcc=40, n_fft=400,
                                 hop_length=160)
    return np.array(mfccs.T.tolist())[..., np.newaxis]


def apply_clean_label(files, y, board, target_label_int, pois_path, pathname,
                      limit, name_part):
    # TODO: pois_path and pathname should be explained
    """
    This is the core function for poisoning in a clean-label style our dataset.
    """
    count = 0
    # A list with x_train, y_train, f_train,
    arrays = [[], [], []]

    for i, f in enumerate(files):
        # Poison data only from the target class
        # The data is mixed so we need to continue checking until we
        # find the required number.
        if y[i] == target_label_int:
            count += 1
            arrays[0].append(poison_style(f, board, pois_path))
            arrays[1].append(target_label_int)
            arrays[2].append(f)

        if count == limit:
            break

    for j, name in enumerate(["x", "y", "f"]):
        name += f"_{name_part}_poisoned"
        arr = np.asarray(arrays[j])
        np.save(f"{pathname}/{name}", arr, allow_pickle=False,
                fix_imports=False)


def apply_dirty_label(files, y, board, target_label_int, pois_path, pathname,
                      limit, name_part):
    """
    This function generates our dirty label data.
    """
    # TODO: pois_path and pathname should be explained
    arrays = [[], [], []]
    for i, f in enumerate(files[:limit]):
        arrays[0].append(poison_style(f, board, pois_path))
        arrays[1].append(target_label_int)
        arrays[2].append(f)

    for j, name in enumerate(["x", "y", "f"]):
        name += f"_{name_part}_poisoned"
        arr = np.asarray(arrays[j])
        np.save(f"{pathname}/{name}", arr, allow_pickle=False,
                fix_imports=False)


def apply_triggers(arrays, algo, target_label, base_dir, orig_path,
                   validation):
    """
    This function generates one directory with the poisoned data we need for
    each style. It generates also the numpy arrays with the MFCCs of each data
    sample contained in this directory.
    """
    (x_train, y_train, f_train, x_val, y_val, f_val, x_test, y_test, f_test,
     mapping) = arrays
    # Retrieve the integer label of the target class.
    target_label_int = np.where(mapping == target_label)[0][0]
    print(f"The index of \"{target_label}\" is {target_label_int}")

    # arrays[0] is x_train
    trojan_samples = round(arrays[0].shape[0] * MAX_POISONING_RATE)
    val_samples = round(trojan_samples * VALIDATION_SIZE)

    # Get all the different boards.
    boards = get_boards()

    # Loop over the different styles
    for i, board in enumerate(boards):

        # Keep only the directory name
        orig_dir = orig_path.split("/")[-1]
        if algo == "clean-label":
            pathname = f"{base_dir}/{orig_dir}/clean_label_{target_label}/"
        elif algo == "dirty-label":
            pathname = f"{base_dir}/{orig_dir}/dirty_label_{target_label}/"
        pathname += f"style{i}"
        print(f"Generating data for {pathname}")

        # Create base directory if it doesn't exist
        create_path(pathname)

        # Continue if everything is there
        if check_poisoned(pathname, validation):
            continue

        # Create a dir for the poisoned wav files (training).
        pois_path = f"{pathname}/poisoned_wav_train/"
        create_path(pois_path)

        # Generate the poisoned training data
        if algo == "clean-label":
            apply_clean_label(f_train, y_train, board, target_label_int,
                              pois_path, pathname, trojan_samples, "train")
        elif algo == "dirty-label":
            apply_dirty_label(f_train, y_train, board, target_label_int,
                              pois_path, pathname, trojan_samples, "train")

        # If validation data needs to be poisoned too
        if validation:
            pois_path = f"{pathname}/poisoned_wav_val/"
            create_path(pois_path)

            if algo == "clean-label":
                apply_clean_label(f_val, y_val, board, target_label_int,
                                  pois_path, pathname, val_samples, "val")
            elif algo == "dirty-label":
                apply_dirty_label(f_val, y_val, board, target_label_int,
                                  pois_path, pathname, val_samples, "val")

        pois_path = f"{pathname}/poisoned_wav_test/"
        create_path(pois_path)
        x_test_pois = []
        # Create a new x_test with everything poisoned this does not depend on
        # the method of poisoning (clean vs dirty label)
        for i, f in enumerate(f_test):
            if i % 100 == 0:
                print(i)
            x_test_pois.append(poison_style(f, board, pois_path))

        # And save the poisoned x_test
        name = "x_test_poisoned"
        arr = np.asarray(x_test_pois)
        np.save(f"{pathname}/{name}", arr, allow_pickle=False,
                fix_imports=False)


def check(prefix, clean_path):
    """Check if all the numpy arrays have been already generated."""
    ret = True
    for suffix in suffixes:
        if not os.path.exists(f"{clean_path}/{prefix}{suffix}.npy"):
            print(f"{clean_path}/{prefix}{suffix}.npy does not exist")
            ret = False

    return ret


def check_poisoned(pathname, validation):
    ret = True
    for prefix in ["x_", "y_", "f_"]:
        suffix = "train_poisoned.npy"
        if not os.path.exists(f"{pathname}/{prefix}{suffix}"):
            print(f"{pathname}/{prefix}{suffix} does not exist")
            ret = False

    if validation:
        for prefix in ["x_", "y_", "f_"]:
            suffix = "val_poisoned.npy"
            if not os.path.exists(f"{pathname}/{prefix}{suffix}"):
                print(f"{pathname}/{prefix}{suffix} does not exist")
                ret = False

    if not os.path.exists(f"{pathname}/x_test_poisoned.npy"):
        print(f"{pathname}/x_test_poisoned.npy does not exist")
        ret = False

    return ret


def load_arrays(prefix, clean_path):
    """Check if all the numpy arrays have been already generated."""
    arrays = []
    for suffix in suffixes:
        arrays.append(np.load(f"{clean_path}/{prefix}{suffix}.npy",
                              fix_imports=False))
    return arrays


if __name__ == "__main__":
    parser = arguments.parser("Choose calculated features")
    args = parser.parse_args()

    # Check if given directory exists.
    if not os.path.isdir(args.path):
        print("Given directory does not exist")
        exit(1)

    # There was functionality for the spectrogram also here that's why we will
    # keep this if statement.
    if args.features == "mfccs":
        # Keep only the dir name instead of the whole path to identify the
        # original dataset
        prefix = (f"mfccs_{args.path.split('/')[-1]}_{args.rate}_"
                  f"{args.n_mfcc}_{args.n_fft}_{args.l_hop}_")

    # Generate the arrays with the required MFCCs of the whole dataset.
    if check(prefix, args.clean_path):
        # Load all the 10 arrays as a tuple
        arrays = load_arrays(prefix, args.clean_path)
    else:
        arrays = preprocess_data(args, prefix)

    # Generate the poisoned data for the clean label
    print(f"Creating folders for clean-label")
    apply_triggers(arrays, "clean-label", args.target_label, args.poisoned_dir,
                   args.path, args.validation)

    # Generate the poisoned data for the dirty label
    print(f"Creating folders for dirty-label")
    apply_triggers(arrays, "dirty-label", args.target_label, args.poisoned_dir,
                   args.path, args.validation)
