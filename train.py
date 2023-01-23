import gc
import os
import math
import copy
import arguments

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics as met
from tensorflow.keras import backend
from prepare_data import load_arrays, create_path

from models import get_model
from styles import get_boards
from tensorflow.keras.backend import clear_session


BATCH_SIZE = 256
PATIENCE = 20


def train(model, epochs, batch_size, patience, x_train, y_train, x_validation,
          y_validation):
    """Trains model

    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't
                           an improvement on accuracy
    :param x_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param x_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set

    :return history: Training history
    """
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience, verbose=1,
                                          restore_best_weights=True)
    # train model
    history = model.fit(x_train, y_train, verbose=2, epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_validation, y_validation),
                        callbacks=[es])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    def save_or_show(save=True, filename="history.png"):
        """Use this function to save the plot"""
        if save:
            fig = plt.gcf()
            fig.set_size_inches((25, 15), forward=False)
            fig.savefig(filename)
        else:
            plt.show()

        plt.close()

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    save_or_show()


def attack_accuracy(model, x_test_poisoned, y_test, target_label_int):
    """Evaluate the backdoor success rate."""
    y_pred = model.predict(x_test_poisoned)

    c = 0
    total = 0
    # TODO: Find a better way to do this with numpy expressions
    for i in range(x_test_poisoned.shape[0]):
        if y_test[i] != target_label_int:
            total += 1
            if np.argmax(y_pred[i]) == target_label_int:
                c += 1

    attack_acc = c * 100.0 / total
    print(f"Attack accuracy: {attack_acc}")
    return attack_acc


def count_errors(model, x_test_poisoned, y_test):
    """
    Count errors for a poisoned set in a clean model.

    We need this function to check whether our triggers generate out of
    distribution samples causing a big misclassification rate in a clean model.
    """
    y_pred = model.predict(x_test_poisoned)

    c = 0
    # TODO: Implement this in a better way with numpy expressions
    for i in range(x_test_poisoned.shape[0]):
        if y_test[i] != np.argmax(y_pred[i]):
            c += 1

    attack_acc = c * 100.0 / x_test_poisoned.shape[0]
    print(f"Attack accuracy: {attack_acc}")
    return attack_acc


def get_path(args, algo, style_id):
    """Load the poisoned x_test_poisoned."""
    pathname = f"{args.poisoned_dir}/{args.path.split('/')[-1]}/"
    if algo == "clean-label":
        pathname += f"clean_label_{args.target_label}/"
    elif algo == "dirty-label":
        pathname += f"dirty_label_{args.target_label}/"
    pathname += f"style{style_id}"
    print(f"Loading data {pathname}")

    return pathname


def load_poisoned(args, algo, style_id):
    """Load all the poisoned data."""
    pathname = get_path(args, algo, style_id)

    poisoned_data = []
    for name_part in ["train", "val"]:
        for name in ["x", "y", "f"]:
            name += f"_{name_part}_poisoned.npy"
            arr = np.load(f"{pathname}/{name}", fix_imports=False)
            poisoned_data.append(arr)

    arr = np.load(f"{pathname}/x_test_poisoned.npy", fix_imports=False)
    poisoned_data.append(arr)

    return poisoned_data


def poison_append(x_train, y_train, f_train, x_val, y_val, f_val,
                  poisoned_data, rate, validation):
    """Append the poisoned data into the training data."""
    # TODO: Remove the duplicate code that is copied-pasted many times here.
    trojan_samples = math.ceil(rate * x_train.shape[0] / 100)

    # Add poisoned training data
    x_train = np.append(x_train, poisoned_data[0][:trojan_samples], axis=0)
    y_train = np.append(y_train, poisoned_data[1][:trojan_samples], axis=0)
    f_train = np.append(f_train, poisoned_data[2][:trojan_samples], axis=0)

    # Shuffle data to simulate a random process.
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    f_train = f_train[perm]

    # Add poisoned data into the validation set.
    if validation:
        x_val = np.append(x_val, poisoned_data[3][:trojan_samples], axis=0)
        y_val = np.append(y_val, poisoned_data[4][:trojan_samples], axis=0)
        f_val = np.append(f_val, poisoned_data[5][:trojan_samples], axis=0)

        # Shuffle validation data
        perm = np.random.permutation(x_val.shape[0])
        x_val = x_val[perm]
        y_val = y_val[perm]
        f_val = f_val[perm]

    return (x_train, y_train, f_train, x_val, y_val, f_val)


def poison_replace(x_train, y_train, f_train, x_val, y_val, f_val,
                   poisoned_data, rate, validation):
    """Replace clean samples with their poisoned counterparts."""
    trojan_samples = math.ceil(rate * x_train.shape[0] / 100)

    for i, f in enumerate(poisoned_data[2][:trojan_samples]):
        idx = np.where(f_train == f)
        x_train[idx] = poisoned_data[0][i]
        y_train[idx] = poisoned_data[1][i]

    # Add poisoned data into the validation set.
    if validation:
        for i, f in enumerate(poisoned_data[5][:trojan_samples]):
            idx = np.where(f_train == f)
            x_val[idx] = poisoned_data[3][i]
            y_val[idx] = poisoned_data[4][i]

    return (x_train, y_train, f_train, x_val, y_val, f_val)


def poison(x_train, y_train, f_train, x_val, y_val, f_val, poisoned_data, rate,
           validation, append=False):
    """Wrapper that chooses the poisoning mechanism."""
    if append:
        return poison_append(x_train, y_train, f_train, x_val, y_val, f_val,
                             poisoned_data, rate, validation)
    else:
        return poison_replace(x_train, y_train, f_train, x_val, y_val, f_val,
                              poisoned_data, rate, validation)


def eval_model(i, clean_data, poisoned_data, epochs, rate, arch, style,
               model_name, algo, args, plots=False, shuffle=False, save=True,
               train_model=True):
    """Train a model and collect metrics."""
    (x_train, y_train, f_train, x_val, y_val, f_val, x_test, y_test, f_test,
     mapping) = copy.deepcopy(clean_data)
    target_label_int = np.where(mapping == args.target_label)[0][0]

    x_test_poisoned = poisoned_data[6]
    if rate != 0:
        (x_train, y_train, f_train, x_val, y_val, f_val) = \
                 poison(x_train, y_train, f_train, x_val, y_val,
                        f_val, poisoned_data, rate, args.validation)
    if train_model:
        # create network
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        classes = len(set(y_train))
        model = get_model(arch, input_shape, classes)

        # train network
        history = train(model, epochs, BATCH_SIZE, PATIENCE, x_train, y_train,
                        x_val, y_val)
        if plots:
            # plot accuracy/loss for training/validation set as a function of
            # the epochs
            plot_history(history)
    else:
        # Load model
        model = tf.keras.models.load_model(f"{args.model_dir}/{model_name}",
                                           custom_objects={"backend": backend})

    if save:
        create_path(args.model_dir)
        model.save(f"{args.model_dir}/{model_name}")

    # evaluate network on test set
    loss, acc = model.evaluate(x_test, y_test, verbose=2)

    # This is needed for the F1 score
    y_pred = model.predict(x_test)
    y_pred = map(np.argmax, y_pred)
    y_pred = np.fromiter(y_pred, dtype=int)
    f1_score = met.f1_score(y_test, y_pred, average='weighted')
    print("\nTest loss: {}, test accuracy: {}".format(loss, 100 * acc))

    if rate > 0:
        attack_acc = attack_accuracy(model, x_test_poisoned, y_test,
                                     target_label_int)
    else:
        # Count the miclassifications of a clean model with poisoned data
        attack_acc = count_errors(model, x_test_poisoned, y_test)

    t = "clean" if (rate == 0) else "trojan"
    e = len(history.history["loss"]) if train_model else 0
    metrics = {"type": t, "accuracy": acc, "attack_accuracy": attack_acc,
               "loss": loss, "epochs": e, "f1": f1_score}

    # Use this function to clear some memory because the OOM steps
    # in after running the experiments for a few times in the cluster.
    clear_session()
    del model
    gc.collect()

    return metrics


def csv_line(metrics, i, arch, algo, style_id, rate):
    """Return one entry for the csv given the generated data."""
    return (f"{i},{arch},{algo},{metrics['type']},{metrics['epochs']},"
            f"{style_id},{rate},{metrics['loss']},{metrics['accuracy']},"
            f"{metrics['attack_accuracy']},{metrics['f1']}\n")


def run_experiments(args):
    """Run all the experiments."""
    # Load the data once to avoid excessive data movement between the hard
    # disk (which is based on nfs in the cluster) and the memory of the machine
    if args.features == "mfccs":
        prefix = (f"mfccs_{args.path.split('/')[-1]}_{args.rate}_{args.n_mfcc}"
                  f"_{args.n_fft}_{args.l_hop}_")
    clean_data = load_arrays(prefix, args.clean_path)

    # Add the first line of the CSV only if the file does not exist or is
    # empty.
    columns = (f"iter,arch,algo,type,epochs,style,rate,loss,accuracy,"
               f"attack_accuracy,f1_score\n")
    if ((not os.path.exists(args.results)) or
            (os.path.getsize(args.results) == 0)):
        with open(args.results, "a") as f:
            f.writelines(columns)

    # Dict to keep the training epochs needed for each model.
    epochs_d = {"large_cnn": 300, "small_cnn": 300, "lstm": 100}

    boards = get_boards()
    train_model = True

    # Use this identifier to avoid mixing models trained with only the 10
    # classes and models trained for the whole dataset
    orig_dir = args.path.split('/')[-1]

    for i in range(5):
        for arch in ["small_cnn", "large_cnn", "lstm"]:
            epochs = epochs_d[arch]
            for algo in ["clean-label", "dirty-label"]:
                for style_id, board in enumerate(boards):

                    # Load poisoned data to avoid unnecessary interactions with
                    # the nfs.
                    poisoned_data = load_poisoned(args, algo, style_id)

                    # Run experiments for clean models (poisoning rate 0%).
                    # Some of the models will be trained many times as without
                    # poisoned data nothing changes as the style is changed.
                    # However, we train them many times to calculate the
                    # miclassification rate of a clean model with poisoned data
                    # for all the combinations of the poisoned data, and to
                    # save all these models with the proper names.
                    # TODO: Decide what the percentage is, the whole data or
                    # just the target class
                    for rate in [0, 0.1, 0.5, 1]:
                        model_name = (f"model_{orig_dir}_{i}_{arch}_{rate}_"
                                      f"style{style_id}_{algo}.h5")
                        if os.path.exists(f"{args.model_dir}/{model_name}"):
                            continue

                        metrics = eval_model(i, clean_data, poisoned_data,
                                             epochs, rate, arch, style_id,
                                             model_name, algo, args,
                                             train_model=train_model)
                        line = csv_line(metrics, i, arch, algo, style_id, rate)

                        # Append the results to the csv
                        with open(args.results, "a") as f:
                            f.writelines(line)


if __name__ == "__main__":
    parser = arguments.parser("Run experiments")

    # Read arguments
    args = parser.parse_args()

    # Run everything
    run_experiments(args)
