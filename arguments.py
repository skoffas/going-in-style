import argparse


def parser(desc):
    """Define here the argument parser to avoid code duplication."""
    # The default values used in "Generalized end-to-end loss for speaker
    # verification.  (ICASSP 2018)" and "Backdoor Attack Against Speaker
    # Verification (ICASSP 2021)" are 40 mel-bands (n_mfcc), step of 10ms
    # (l_hop) and a window length of 25ms (n_fft). In this script every value
    # is given as a number of vector elements. E.g. for a 1-sec waveform
    # sampled at 16k we have 16000 vector elements when loaded in the memory.
    # Thus these values will be l_hop = 10*10^{-3} * 16000 = 160 elements,
    # n_fft = 25*10^{-3}*16000 = 400 elements.
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("features", choices=["mfccs"], type=str,
                        help="Choose calculated features")
    parser.add_argument("path", type=str, help="Give the dataset's path")
    parser.add_argument("--clean-path", type=str, help="Give the"
                        "directory that the clean data will be saved as numpy"
                        "arrays", default="./clean_data", nargs="?")
    parser.add_argument("rate", type=int, help="Sampling rate in Hz",
                        default=16000, nargs="?")
    parser.add_argument("n_mfcc", type=int, help="Number of mel-bands",
                        default=40, nargs="?")
    parser.add_argument("n_fft", type=int, help="FFT's window size for the "
                        "mel-spectrogram", default=400, nargs="?")
    parser.add_argument("l_hop", type=int, help="Number of samples between "
                        "successive frames", default=160, nargs="?")
    parser.add_argument("--target-label", type=str, help="The target class for"
                        "the backdoored data", default="yes", nargs="?")
    parser.add_argument("--poisoned-dir", type=str, help="The base directory"
                        "for the poisoned data", default="./poisoned_data",
                        nargs="?")
    # Poison also the validation data with the same poisoning percentage.
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--no-validation', dest='validation',
                        action='store_false')
    parser.set_defaults(shuffle=False)
    # This variable declares if we are going to shuffle the data after it is
    # being poisoned to avoid having all the poisoned data in the beginning of
    # the dataset. It seems that there is no difference in the result after
    # shuffle but we keep this functionality as we may need it in the future.
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.set_defaults(shuffle=False)
    # The CSV that contains all the results.
    parser.add_argument("--results", type=str, help="The name of the csv that"
                        "the data is saved", default="./results.csv",
                        nargs="?")
    parser.add_argument("--model-dir", type=str, help="The name of the"
                        "folder that the models are saved",
                        default="./models", nargs="?")
    return parser
