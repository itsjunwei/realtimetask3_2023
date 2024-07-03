import librosa
import numpy as np
import time
from rich.progress import track
import gc
from all_models import RNet14, CNN8
import torch

gc.enable()
gc.collect()

NFFT = 512
FS = 24000
HOP_LEN = 300
WIN_LEN = 512
MEL_BINS = 128
N_ATTEMPTS = 200
N_SECS = 1
duration = int(N_SECS * FS)

# Mel Filter Variables
melW = librosa.filters.mel(sr=FS, n_fft=NFFT, n_mels=MEL_BINS, fmin=50, fmax=None)
_eps = 1e-8

# Generic SALSA-Lite Variables
_c = 343
_delta = 2 * np.pi * FS / (NFFT * _c)
d_max = 42/1000
fmax_doa = 4000
fmin_doa = 50
fmax = 9000
n_bins = NFFT // 2 + 1
lower_bin = int(np.floor(fmin_doa * NFFT / float(FS))) # 1
upper_bin = int(np.floor(fmax_doa * NFFT / float(FS))) # 42
lower_bin = np.max((1, lower_bin))
cutoff_bin = int(np.floor(fmax * NFFT / float(FS)))
freq_vector = np.arange(n_bins)
freq_vector[0] = 1
freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1

def extract_salsalite(audio_data):
    """
    The audio data is from the ambimik, assuming that it is a four-channel array
    The shape should be (4 , x) for (n_channels, time*fs)
    """
    log_specs = []

    for imic in np.arange(audio_data.shape[0]):
        stft = librosa.stft(y=np.asfortranarray(audio_data[imic]), 
                            n_fft=NFFT, 
                            hop_length=HOP_LEN,
                            center=True, 
                            window='hann', 
                            pad_mode='reflect')

        if imic == 0:
            n_frames = stft.shape[1]
            X = np.zeros((n_bins, n_frames, audio_data.shape[0]), dtype='complex')  # (n_bins, n_frames, n_mics)
        X[:, :, imic] = stft
        # Compute log linear power spectrum
        spec = (np.abs(stft) ** 2).T
        log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
        log_spec = np.expand_dims(log_spec, axis=0)
        log_specs.append(log_spec)
    log_specs = np.concatenate(log_specs, axis=0)  # (n_mics, n_frames, n_bins)

    # Compute spatial feature
    # X : (n_bins, n_frames, n_mics) , NIPD formula : -(c / (2pi x f)) x arg[X1*(t,f) . X2:M(t,f)]
    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
    phase_vector = phase_vector / (_delta * freq_vector)
    phase_vector = np.transpose(phase_vector, (2, 1, 0))  # (n_mics, n_frames, n_bins)

    # Crop frequency
    log_specs = log_specs[:, :, lower_bin:cutoff_bin]
    phase_vector = phase_vector[:, :, lower_bin:cutoff_bin]
    phase_vector[:, :, upper_bin:] = 0

    # Stack features
    audio_feature = np.concatenate((log_specs, phase_vector), axis=0)
    
    return audio_feature

def gcc_phat(sig, refsig) -> np.ndarray:
    """
    Compute GCC-PHAT between sig and refsig.
    :param sig: <np.ndarray: (n_samples,).
    :param refsig: <np.ndarray: (n_samples,).
    :return: gcc_phat: <np.ndarray: (1, n_frames, n_mels)>
    """
    ncorr = 2 * NFFT - 1
    n_fft = int(2 ** np.ceil(np.log2(np.abs(ncorr))))  # this n_fft double the length of win_length
    Px = librosa.stft(y=np.asfortranarray(sig),
                        n_fft=n_fft,
                        hop_length=HOP_LEN,
                        win_length=WIN_LEN,
                        center=True,
                        window="hann",
                        pad_mode='reflect')
    Px_ref = librosa.stft(y=np.asfortranarray(refsig),
                            n_fft=n_fft,
                            hop_length=HOP_LEN,
                            win_length=WIN_LEN,
                            center=True,
                            window="hann",
                            pad_mode='reflect')
    # Filter gcc spectrum, cutoff frequency = 4000Hz, buffer bandwidth = 400Hz
    freq_filter = np.ones((n_fft//2 + 1, 1))
    k_cutoff = int(4000 / FS * n_fft)
    k_buffer = int(400 / FS * n_fft)
    cos_x = np.arange(2 * k_buffer) * (np.pi/2) / (2 * k_buffer - 1)
    freq_filter[k_cutoff - k_buffer: k_cutoff + k_buffer, 0] = np.cos(cos_x)
    Px = Px * freq_filter
    Px_ref = Px_ref * freq_filter

    R = Px * np.conj(Px_ref)
    n_frames = R.shape[1]
    gcc_phat = []
    for i in range(n_frames):
        spec = R[:, i].flatten()
        cc = np.fft.irfft(np.exp(1.j * np.angle(spec)))
        cc = np.concatenate((cc[-MEL_BINS // 2:], cc[:MEL_BINS // 2]))
        gcc_phat.append(cc)
    gcc_phat = np.array(gcc_phat)
    gcc_phat = gcc_phat[None, :, :]

    return gcc_phat


def extract_logmel_gccphat(audio_input):
    n_channels = audio_input.shape[0]
    features = []
    gcc_features = []
    for n in range(n_channels):
        
        spec = np.abs(librosa.stft(y=np.asfortranarray(audio_input[n]),
                                   n_fft=NFFT,
                                   hop_length=HOP_LEN,
                                   win_length=WIN_LEN,
                                   center=True,
                                   window="hann",
                                   pad_mode="reflect"))
        
        mel_spec = np.dot(melW, spec ** 2).T
        logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
        logmel_spec = np.expand_dims(logmel_spec, axis=0)
        
        features.append(logmel_spec)
        for m in range(n + 1, n_channels):
            gcc_features.append(gcc_phat(sig=audio_input[m], refsig=audio_input[n]))

    features.extend(gcc_features)
    features = np.concatenate(features, axis=0)
    
    return features



def extract_melIV(audio_input: np.ndarray) -> np.ndarray:
    """
    :param audio_input: <np.ndarray: (4, n_samples)>.
    :return: logmel <np.ndarray: (7, n_timeframes, n_features)>.
    """
    n_channels = audio_input.shape[0]
    features = []
    X = []

    for i_channel in range(n_channels):
        spec = librosa.stft(y=np.asfortranarray(audio_input[i_channel]),
                            n_fft=NFFT,
                            hop_length=HOP_LEN,
                            win_length=WIN_LEN,
                            center=True,
                            window="hann",
                            pad_mode='reflect')
        X.append(np.expand_dims(spec, axis=0))  # 1 x n_bins x n_frames

        # compute logmel
        mel_spec = np.dot(melW, np.abs(spec) ** 2).T
        logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
        logmel_spec = np.expand_dims(logmel_spec, axis=0)
        features.append(logmel_spec)

    # compute intensity vector: for ambisonic signal, n_channels = 4
    X = np.concatenate(X, axis=0)  # 4 x n_bins x n_frames
    IVx = np.real(np.conj(X[0, :, :]) * X[1, :, :])
    IVy = np.real(np.conj(X[0, :, :]) * X[2, :, :])
    IVz = np.real(np.conj(X[0, :, :]) * X[3, :, :])

    normal = np.sqrt(IVx ** 2 + IVy ** 2 + IVz ** 2) + _eps
    IVx = np.dot(melW, IVx / normal).T  # n_frames x n_mels
    IVy = np.dot(melW, IVy / normal).T
    IVz = np.dot(melW, IVz / normal).T

    # add intensity vector to logmel
    features.append(np.expand_dims(IVx, axis=0))
    features.append(np.expand_dims(IVy, axis=0))
    features.append(np.expand_dims(IVz, axis=0))
    feature = np.concatenate(features, axis=0)

    return feature

if __name__ == "__main__":

    common_output_shape = (1, 10, 117)
    gcc_inshape = (1, 10, 81, 128)
    iv_inshape = (1, 7, 81, 128)
    sl_inshape = (1, 7, 81, 191)
    

    """
    MEL-GCCPHAT FEATURE EXTRACTION AND INFERENCE TIME CHECKS
    """

    model = RNet14(input_shape=gcc_inshape,
                   output_shape=common_output_shape).to(torch.device("cpu"))
    # model = CNN8(in_feat_shape=gcc_inshape,
    #              out_shape=common_output_shape).to(torch.device("cpu"))
    model.eval()

    melGCC_time = []
    GCC_inference = []
    for _ in track(range(N_ATTEMPTS), description="Extracting MelSpec GCCPHAT..."):
        random_input = np.random.rand(4,duration)

        start_time = time.time()
        melGCC = extract_logmel_gccphat(random_input)
        end_time = time.time()
        
        infer_start = time.time()
        with torch.no_grad():
            melGCC = np.expand_dims(melGCC, axis=0)
            x = torch.from_numpy(melGCC).to(torch.device("cpu")).float()
            output = model(x)
        infer_end = time.time()

        melGCC_time.append(end_time-start_time)
        GCC_inference.append(infer_end - infer_start)

    melGCC_time = np.array(melGCC_time)
    print("[MelSpec GCC] Extraction Time ~ N({:0.3f}, {:0.3f})".format(np.mean(melGCC_time), np.var(melGCC_time)))
    
    GCC_inference = np.array(GCC_inference)
    print("[MelSpec GCC] Inference Time ~ N({:0.3f}, {:0.3f})".format(np.mean(GCC_inference), np.var(GCC_inference)))
    
    gc.enable()
    gc.collect()

    """MEL-IV Feature Extraction and Inference benchmarking"""
    
    model = RNet14(input_shape=iv_inshape, output_shape=common_output_shape).to(torch.device("cpu"))
    model.eval()

    melIV_times = []
    iv_inference = []
    for _ in track(range(N_ATTEMPTS), description='Extracting MelSpecIVs...'):
        random_input = np.random.rand(4,duration)

        start_time = time.time()
        melIV = extract_melIV(random_input)
        end_time = time.time()
        
        infer_start = time.time()
        with torch.no_grad():
            melIV = np.expand_dims(melIV, axis=0)
            x = torch.from_numpy(melIV).to(torch.device("cpu")).float()
            output = model(x)
        infer_end = time.time()

        melIV_times.append(end_time-start_time)
        iv_inference.append(infer_end - infer_start)

    melIV_times = np.array(melIV_times)
    print("[MelSpec IV] Extraction Time ~ N({:0.3f}, {:0.3f})".format(np.mean(melIV_times), np.var(melIV_times)))
    
    iv_inference = np.array(iv_inference)
    print("[MelSpec IV] Inference Time ~ N({:0.3f}, {:0.3f})".format(np.mean(iv_inference), np.var(iv_inference)))

    gc.enable()
    gc.collect()

    """
    SALSALITE FEATURE EXTRACTION AND INFERENCE TIME CHECKS
    """

    model = RNet14(input_shape=sl_inshape,
                   output_shape=common_output_shape).to(torch.device("cpu"))
    model.eval()

    salsalite_time = []
    salsalite_inference_time = []
    for _ in track(range(N_ATTEMPTS), description='Extracting SALSA-Lite...'):
        random_input = np.random.rand(4,duration)

        start_time = time.time()
        salsalite = extract_salsalite(random_input)
        end_time = time.time()

        infer_start = time.time()
        with torch.no_grad():
            salsalite = np.expand_dims(salsalite, axis=0)
            x = torch.from_numpy(salsalite).to(torch.device("cpu")).float()
            output = model(x)
        infer_end = time.time()

        salsalite_time.append(end_time-start_time)
        salsalite_inference_time.append(infer_end - infer_start)

    salsalite_time = np.array(salsalite_time)
    print("[SALSA-Lite] Extraction Time ~ N({:0.3f}, {:0.3f})".format(np.mean(salsalite_time), np.var(salsalite_time)))
    
    salsalite_inference_time = np.array(salsalite_inference_time)
    print("[SALSA-Lite] Inference Time ~ N({:0.3f}, {:0.3f})".format(np.mean(salsalite_inference_time), np.var(salsalite_inference_time)))