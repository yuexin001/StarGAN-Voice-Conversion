import argparse
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from model import StarGANVC
from preprocess import *
from utility import *

# get all speaker
all_speaker = get_speakers(trainset='./data/fourspeakers')
label_enc = LabelEncoder()
label_enc.fit(all_speaker)


def conversion(model_dir, test_dir, output_dir, source, target):
    if not os.path.exists(model_dir) or not os.path.exists(test_dir):
        raise Exception('model dir or test dir not exist!')
    model = StarGANVC(num_features=FEATURE_DIM, mode='test')

    model.load(filepath=os.path.join(model_dir, MODEL_NAME))
    p = os.path.join(test_dir, f'{source}/*.wav')
    tempfiles = glob.glob(p)
    normlizer = Normalizer()

    for one_file in tempfiles:
        _, speaker, name = os.path.normpath(one_file).rsplit(os.sep, maxsplit=2)
        wav_, fs = librosa.load(one_file, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
        wav, pad_length = pad_wav_to_get_fixed_frames(wav_, frames=FRAMES)

        f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=500.0)

        sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=FFTSIZE)

        ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=FFTSIZE)
        coded_sp = pyworld.code_spectral_envelope(sp, fs, FEATURE_DIM)

        coded_sps_mean = np.mean(coded_sp, axis=0, dtype=np.float64, keepdims=True)
        coded_sps_std = np.std(coded_sp, axis=0, dtype=np.float64, keepdims=True)

        one_test_sample = []
        csp_transpose = coded_sp.T  # 36x512 36x128...
        for i in range(0, csp_transpose.shape[1] - FRAMES + 1, FRAMES):
            t = csp_transpose[:, i:i + FRAMES]
            t = normlizer.forward_process(t, speaker)
            t = np.reshape(t, [t.shape[0], t.shape[1], 1])
            one_test_sample.append(t)

        one_test_sample_label = np.zeros([len(one_test_sample), len(all_speaker)])
        temp_index = label_enc.transform([target])[0]
        one_test_sample_label[:, temp_index] = 1

        generated_results = model.test(one_test_sample, one_test_sample_label)

        reshpaped_res = []
        for one in generated_results:
            t = np.reshape(one, [one.shape[0], one.shape[1]])

            t = normlizer.backward_process(t, target)
            reshpaped_res.append(t)
        c = []
        for one_slice in reshpaped_res:
            one_slice = np.ascontiguousarray(one_slice.T, dtype=np.float64)
            decoded_sp = pyworld.decode_spectral_envelope(one_slice, SAMPLE_RATE, fft_size=FFTSIZE)
            c.append(decoded_sp)

        concated = np.concatenate((c), axis=0)
        f0 = normlizer.pitch_conversion(f0, speaker, target)

        synwav = pyworld.synthesize(f0, concated, ap, fs)
        synwav = synwav[:-pad_length]

        wavname = f'{speaker}-{target}+{name}'
        wavpath = f'{output_dir}/wavs'
        if not os.path.exists(wavpath):
            os.makedirs(wavpath, exist_ok=True)
        librosa.output.write_wav(f'{wavpath}/{wavname}', synwav, sr=fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting.')

    model_dir = './out/model/'
    test_dir = './data/fourspeakers_test/'
    source_speaker = 'SF1'
    target_speaker = 'TM1'
    output_dir = './converted_voices'

    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--test_dir', type=str, default=test_dir)
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--source_speaker', type=str, default=source_speaker)
    parser.add_argument('--target_speaker', type=str, default=target_speaker)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    test_dir = argv.test_dir
    output_dir = argv.output_dir
    source_speaker = argv.source_speaker
    target_speaker = argv.target_speaker

    conversion(model_dir, test_dir, output_dir, source_speaker, target_speaker)
