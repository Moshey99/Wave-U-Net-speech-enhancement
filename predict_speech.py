import argparse
import os
from sre_constants import SRE_INFO_PREFIX

import data.utils
import model.utils as model_utils
import numpy as np
import math

from test import predict_song
from model.waveunet import Waveunet
from data.utils import resample

def main(args):
    # path
    folder_path = args.folder
    total_SNR = 0
    length = len(os.listdir(folder_path))
    for i in range(1,length+1):
      input_path = folder_path + '/audio_'+str(i)+'/mixture.wav'
      clean_path = folder_path + '/audio_'+str(i)+'/vocals.wav'

      # MODEL
      num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                    [args.features*2**i for i in range(0, args.levels)]
      target_outputs = int(args.output_size * args.sr)
      model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                      target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                      conv_type=args.conv_type, res=args.res, separate=args.separate)

      if args.cuda:
          model = model_utils.DataParallel(model)
          print("move model to gpu")
          model.cuda()

      state = model_utils.load_model(model, None, args.load_model, args.cuda)
      preds = predict_song(args, input_path, model)

      result_speech = preds['vocals']
      result_noise = preds['other']

      orig_speech,orig_sr = data.utils.load(clean_path, sr=16000, mono=False)
      orig_speech_44 = orig_speech
      orig_speech_44 = resample(orig_speech_44, orig_sr, 44100)

      # orig_mix,orig_sr = data.utils.load(args.input, sr=16000, mono=False)
      # orig_mix_44 = orig_mix
      # orig_mix_44 = resample(orig_mix_44, orig_sr, 44100)

      est_result_noise = result_speech - orig_speech_44

      # est_orig_noise = orig_mix - orig_speech

      curr_SNR = calc_SNR(orig_speech_44,est_result_noise)
      total_SNR = total_SNR + curr_SNR
    
    mean_SNR = total_SNR/length ; 
    print('THE MEAN SNR IS: ',mean_SNR)
      




def calc_SNR(speech,noise):


  # Calculate the power of the speech and noise signals
  power_noise = (np.mean(noise ** 2))
  # print(power_noise)

  # Detect non-silent segments in the speech audio
  silence_threshold = 0.0093  # Adjust this threshold as needed
  non_silent_indices = np.where(speech ** 2 > silence_threshold)

  # Calculate the power of non-silent speech segments
  power_non_silent_speech = np.mean(speech[non_silent_indices] ** 2)
  # print(power_non_silent_speech)

  # Calculate SNR in dB
  snr = 10 * np.log10(power_non_silent_speech / power_noise)
  print(snr)

  return snr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--load_model', type=str, default='checkpoints/waveunet/model',
                        help='Reload a previously trained model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=2,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--folder', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to folder")
    args = parser.parse_args()

    main(args)
