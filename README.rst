Jingle-Back: Backdoor Attacks with Stylistic Transformations
============================================================
In this project we implement a backdoor attack in speech recognition with
stylistic transformations as our trigger. For our stylistic transformations we
used `Spotify's Pedalboard <https://github.com/spotify/pedalboard>`_ and we
named our attack Jingle-Back.

Guide
-----
First a `virtual environment
<https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_
has to be created, and activated. Then the requirements should be installed::

  $ python -m pip install -r requirements.txt

To download the speech commands dataset and calculate features for both the
clean and the poisoned samples the script ``startup.sh`` should be run::

  $ ./startup.sh

This script selects only 10 classes from the whole dataset to demonstrate the
functionality faster in less data. For the data preprocessing
``prepare_data.py`` is used.

After preparing the features, the models can be trained with the following
command::

  $ python train.py mfccs data_10

About
-----
This is the repo for our paper "Going in Style: Audio Backdoors Through
Stylistic Transformations" which was presented in ICASSP 2023. To reference our
paper use the following bibtex entry::

  @inproceedings{koffas2023going,
    title={Going In Style: Audio Backdoors Through Stylistic Transformations},
    author={Koffas, Stefanos and Pajola, Luca and Picek, Stjepan and Conti, Mauro},
    booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={1--5},
    year={2023},
    organization={IEEE}
  }
