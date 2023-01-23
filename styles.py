"""
A file that provides the functionality needed to poison our data with different
styles.
"""
from pedalboard.io import AudioFile
from pedalboard import LadderFilter, PitchShift, Gain, Phaser
from pedalboard import Pedalboard, Chorus, Reverb, Distortion


def get_boards():
    """Return all the styles we are going to use."""
    boards = []

    ## Semitone (style 0)
    pedal = PitchShift(semitones=10)
    board = Pedalboard([pedal])
    boards.append(board)

    ## Distortion (style 1)
    pedal = Distortion(drive_db=30)
    board = Pedalboard([pedal])
    boards.append(board)

    ## chorus (style 2)
    pedal = Chorus(rate_hz=1, depth=5, centre_delay_ms=10.0, feedback=0.0,
                   mix=0.5)
    board = Pedalboard([pedal])
    boards.append(board)

    ## multi 1 (style 3)
    pedal1 = PitchShift(semitones=10)
    pedal2 = Distortion(drive_db=20)
    pedal3 = Chorus(rate_hz=1, depth=5, centre_delay_ms=8.0, feedback=0.0,
                    mix=0.5)
    board = Pedalboard([pedal1, pedal2, pedal3])
    boards.append(board)

    ## multi 2 (style 4)
    board = Pedalboard([Chorus(centre_delay_ms=15), Distortion(20),
                        Reverb(room_size=0.6)])
    boards.append(board)

    ## multi 3 (style 5)
    board = Pedalboard([
        Gain(gain_db=12),
        LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=1000),
        Phaser(),
    ])
    boards.append(board)

    return boards
