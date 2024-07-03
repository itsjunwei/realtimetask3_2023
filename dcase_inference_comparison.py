import numpy as np
import pyaudio
import threading
import torch
from datetime import datetime
from collections import deque
import time
import os 
import gc
import warnings
from comparing_feature_extraction import extract_logmel_gccphat, extract_melIV, extract_salsalite
from all_models import RNet14, CNN8, Baseline, ResNet18, CNN4


# For testing
extraction_time = []

#suppress warnings for numpy overflow encountered in exp function during sigmoid
warnings.filterwarnings('ignore')

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.system('cls')
print("Changing directory to : ", dname)
print("Screen cleared!")
gc.enable()
gc.collect()
n_classes = 3

# Global variables
CHUNK = 500
FORMAT = pyaudio.paFloat32
CHANNELS = 4
RATE = 24000
MAX_RECORDINGS = 48
INPUT_DEVICE_INDEX = 1
RECORD_SECONDS = 1
fpb = int(RATE * RECORD_SECONDS) # Frames per buffer

common_output_shape = (1, 10, 117)
gcc_inshape = (1, 10, 81, 128)
iv_inshape = (1, 7, 81, 128)
sl_inshape = (1, 7, 81, 191)

# Queue to store audio buffers 
data_queue = deque()
feat_extraction_times = []
inference_times = []
n_times = 0

# Stream
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=fpb,
                    input_device_index=INPUT_DEVICE_INDEX)

lock = threading.Lock()

# Function to record audio
def record_audio(stream, stop_event):
    """Define the data buffer queue outside of the function and call it globally.
    This function is used to record audio data and push them to the 
    buffer queue for another function to use for inference. 
    
    Inputs:
        stream (pyaudio.stream) : Stream class defined by PyAudio
        stop_event (thread) : Thread to indicate whether this thread should continue running
        
    Returns:
        None
    """

    global data_queue # Bufer queue for all the data

    while not stop_event.is_set():
        try:
            # Read the data from the buffer as np.float32
            time_now = datetime.now()
            buffer = np.frombuffer(stream.read(fpb, exception_on_overflow=False), dtype=np.float32)

            # Append the audio data and recording time into the buffer queues
            data_queue.append((time_now, buffer))
        except Exception as e:
            print("Something went wrong!")
            print(e)
            break
            

def infer_save_audio(model):
    """Define the data buffer queue outside of the function. In fact, may not even need to 
    append the recording time into the buffer if do not wish to keep track of time. It was
    used to keep track of when the audio data was recorded and to make sure that the system
    is inferring on the correct audio data in the queue.
    
    This function is used to use an ONNXRUNTIME session in order to infer on the audio data
    recorded. Potentially, this function can be modified to return the inference output itself
    to pass to another function/system for post-processing.
    
    Inputs:
        ort_sess (onnxruntime session) : The onnxruntime session of our model for inference
        
    Returns:
        None
    """

    global data_queue, n_times

    # Wait until there is something in the buffer queue
    while len(data_queue) == 0:
        pass

    # We take the latest (most recent) item and copy it in order to make modifications on the data
    all_data = data_queue.popleft()
    record_time = all_data[0] # No need copy for string data, apparently
    audio_data = all_data[1].copy() # Float data needs to be copied
    audio_data = audio_data.reshape(-1,4).T

    # Feature extraction
    feat_start = time.time()
    features = extract_salsalite(audio_data)
    # features = extract_logmel_gccphat(audio_data)
    # features = extract_melIV(audio_data)

    features = np.expand_dims(features, axis=0)
    x = torch.from_numpy(features).to(torch.device("cpu")).float()

    feat_end = time.time()
    feat_extraction_times.append(feat_end - feat_start)
    
    # Model prediction
    pred_start = time.time()
    
    output = model(x)
    pred_end = time.time()
    inference_times.append(pred_end - pred_start)
    
    # Just a verbose message
    n_times += 1
    print("[{}] -- {} times -- {} / {}".format(record_time.strftime("%H:%M:%S.%f")[:-3], n_times, features.shape, output.shape))


def main():
    """Main function to do concurrent recording and inference"""
    
    # model = Baseline(input_shape = sl_inshape,
    #                  output_shape= common_output_shape).to(torch.device("cpu"))
    # model = RNet14(input_shape=gcc_inshape,
    #                output_shape=common_output_shape, use_conformer=True).to(torch.device("cpu"))
    # model = ResNet18(input_shape = gcc_inshape,
    #                 output_shape=common_output_shape,
    #                 use_selayers=True).to(torch.device("cpu"))
    # model = CNN8(in_feat_shape=gcc_inshape, 
    #              out_shape=common_output_shape).to(torch.device("cpu"))
    model = CNN4(in_feat_shape=sl_inshape, 
                 out_shape=common_output_shape).to(torch.device("cpu"))
    model.eval()
    print(model)

    # Create an event to signal the threads to stop
    stop_event = threading.Event()
    record_thread = threading.Thread(target=record_audio,
                                    args=(stream,stop_event))
    record_thread.start()
    print("Threads started!")

    try:
        while True:
            infer_save_audio(model)
    except KeyboardInterrupt:
        # Signal the threads to stop
        stop_event.set()

        # Wait for the threads to finish
        record_thread.join()
        print("Recording stopped by user")
        
        # End the stream gracefully
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
 
 
    # Main recording and inference function
    main()
    
    print("Feature Extraction ~ N({:.4f}, {:.4f})".format(np.mean(np.array(feat_extraction_times)),
                                                          np.var(np.array(feat_extraction_times))))
    
    print("Model Inference ~ N({:.4f}, {:.4f})".format(np.mean(np.array(inference_times)),
                                                          np.var(np.array(inference_times))))
