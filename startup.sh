base_dir=$PWD
FILE="speech_commands_v0.01.tar.gz"

if test -f "$FILE"; then
	echo "$FILE exists."
else
	# Download the data
	wget download.tensorflow.org/data/speech_commands_v0.01.tar.gz
fi

# Extract the full dataset
mkdir data
tar xzf speech_commands_v0.01.tar.gz --directory data

# Create a directory with the 10 classes only to have something smaller to play
# with
mkdir data_10

cd data_10
for dir in yes no up down left right on off stop go;
do
	#ln -s $base_dir/data/$dir .
	cp -r $base_dir/data/$dir .
done

cd -

# Preprocess data
python prepare_data.py mfccs data_10 --validation
