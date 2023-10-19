This is the folder of the Wave-U-Net fine-tuned model for speech enhancement, implemented in PyTorch.


Contents of the folder:
Waveunet_training.ipynb - trains Wave-U-net
Fine_tuning_upsmaple_model.ipynb - trains and evaluates the decoder-learned variant only
Running_train_evaluation_other_models.ipynb - trains and evaluates all the other variants, as well as evaluates the different datasets.
Requirements.txt - the required packages that need to be installed 
Checkpoints - contains the trained weights of the variants of the fine-tuned models and the weights of the original Wave-U-Net. Used to contain checkpoints of the model during training.
Model - contains the architecture of the model.
hdf -  to save the preprocessed data (HDF files)
MS-SNSD-master - a folder that contains what is needed to create different datasets.
MUSDB18_format_of_SNSD - dataset that the variants were trained and mainly evaluated on, in a format that the model is able to process. Contains various noise types on 20dB level.
MUSDB_format_noises_evaluation - dataset to evaluate different noise types
MUSDB_format_dB_evaluation - dataset to evaluate different noise levels
Audio_examples - some examples of the outputs of the model. Contains vocal evaluation of the different variant
Train.py - the main function to run all the training process
Predict.py - the main function to run trained model on an audio file
Predict_speech.py - produces the SNR of the model on a given dataset


Installation
Install all the required packages listed in the requirements.txt:
----------------------------------------
pip install -r requirements.txt
----------------------------------------


How to train
Basically, all needed to do to train the model on a dataset is to use this command:
---------------------------------------------------------------------------------
python train.py --dataset_dir ‘MUSDB18_format_of_SNSD’
---------------------------------------------------------------------------------
Where dataset_dir is a configuration argument that must contain the directory of dataset’s folder. 
There are many more config arguments, such as --cuda for GPU activation, or --load_model to start training with loaded weights instead of initial ones. All the arguments are in train.py.


How to test the model on a song
To apply the model on a song, use the command below:
--------------------------------------------------------------------------------------------------------------------------
python predict.py --load_model ‘checkpoints/all_params_model’ --input ‘PATH_OF_SONG’
--------------------------------------------------------------------------------------------------------------------------


--input points to the music file that the user is willing to separate
--cuda activates the GPU
The output will be written to the parent folder of the music file, unless the output directory is defined by the argument --output


How to evaluate model’s enhancement performance on dataset:
To evaluate the model on a dataset and produce its enhanced SNR, use the command below:
------------------------------------------------------------------------------------------------
python 'predict_speech.py' --load_model 'checkpoints/all_params_model' --folder 'DATASET_FOLDER'
------------------------------------------------------------------------------------------------