# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        # INPUT PATH
        dataset_dir='DCASE2023_SELD_dataset/raw_data',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATHS
        feat_label_dir='DCASE2023_SELD_dataset/seld_feat_label/',  # Directory to dump extracted features and labels
 
        model_dir='models/',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals
        unique_classes = 13,

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.0125,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 4000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        use_augmentations=False,
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.
        use_resnet = False,
        use_conformer=False,

        # DNN MODEL PARAMETERS
        label_sequence_length=10,    # Feature sequence length
        batch_size=32,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,
        final_lr = 1e-5,

        # METRIC
        average='macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,
        normalize_specs = True
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("MIC + SALSA + multi ACCDOA + ResNet-GRU \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['use_augmentations'] = True
        params['use_conformer'] = False
        params['use_resnet'] = True

    elif argv == '2':
        print("MIC + SALSA + multi ACCDOA + ResNet-Conformer \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['use_augmentations'] = True
        params['use_conformer'] = True
        params['use_resnet'] = True

    elif argv == '3':
        print("MIC + SALSA + multi ACCDOA + Baseline + No Augs \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['use_augmentations'] = False
        params['use_conformer'] = False
        params['use_resnet'] = False

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached



    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
