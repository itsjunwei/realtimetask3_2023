from torch.optim.lr_scheduler import _LRScheduler
import warnings
import numpy as np


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, milestones: tuple = (0.05, 0.8), min_lr: float = 1e-5, last_epoch: int=-1):

        self.warmup_epochs = int(milestones[0] * total_epochs)
        self.maintain_epochs = int(milestones[1] * total_epochs)
        self.decay_epochs = total_epochs - self.warmup_epochs - self.maintain_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            return [(base_lr / self.warmup_epochs) * (epoch + 1) for base_lr in self.base_lrs]
        elif epoch < self.warmup_epochs + self.maintain_epochs:
            # Maintain learning rate
            return [base_lr for base_lr in self.base_lrs]
        else:
            # Linear decay
            decay_epoch = epoch - self.warmup_epochs - self.maintain_epochs
            decay_factor = 1 - decay_epoch / self.decay_epochs
            return [base_lr * decay_factor + self.min_lr * (1 - decay_factor) for base_lr in self.base_lrs]


class ComposeTransformNp:
    """
    Compose a list of data augmentation on numpy array.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x


class DataAugmentNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError

class RandomShiftUpDownNp(DataAugmentNumpyBase):
    """
    This data augmentation random shift the spectrogram up or down.
    """
    def __init__(self, always_apply=False, p=0.5, freq_shift_range: int = None, direction: str = None, mode='reflect',
                 n_last_channels: int = 0):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.n_last_channels = n_last_channels

    def apply(self, x: np.ndarray):
        if self.always_apply is False:
            return x
        else:
            if np.random.rand() < self.p:
                return x
            else:
                n_channels, n_timesteps, n_features = x.shape
                if self.freq_shift_range is None:
                    self.freq_shift_range = int(n_features * 0.08)
                shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
                if self.direction is None:
                    direction = np.random.choice(['up', 'down'], 1)[0]
                else:
                    direction = self.direction
                new_spec = x.copy()
                if self.n_last_channels == 0:
                    if direction == 'up':
                        new_spec = np.pad(new_spec, ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
                    else:
                        new_spec = np.pad(new_spec, ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
                else:
                    if direction == 'up':
                        new_spec[:-self.n_last_channels] = np.pad(
                            new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
                    else:
                        new_spec[:-self.n_last_channels] = np.pad(
                            new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
                return new_spec
            

class CompositeCutout(DataAugmentNumpyBase):
    """
    This data augmentation combine Random cutout, specaugment, cutout hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_cutout = RandomCutoutNp(always_apply=True, image_aspect_ratio=image_aspect_ratio,
                                            n_zero_channels=n_zero_channels,
                                            is_filled_last_channels=is_filled_last_channels)
        self.spec_augment = SpecAugmentNp(always_apply=True, n_zero_channels=n_zero_channels,
                                          is_filled_last_channels=is_filled_last_channels)
        self.random_cutout_hole = RandomCutoutHoleNp(always_apply=True, n_zero_channels=n_zero_channels,
                                                     is_filled_last_channels=is_filled_last_channels)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 3, 1)[0]
        if choice == 0:
            return self.random_cutout(x)
        elif choice == 1:
            return self.spec_augment(x)
        elif choice == 2:
            return self.random_cutout_hole(x)

class RandomCutoutNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a rectangular area from the input image. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 random_value: float = None, n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param image_aspect_ratio: height/width ratio. For spectrogram: n_time_steps/ n_features.
        :param random_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_value = random_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels
        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features) or (n_time_steps, n_features)>: input spectrogram.
        :return: random cutout x
        """
        # get image size
        image_dim = x.ndim
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        # Initialize output
        output_img = x.copy()
        # random erase
        s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
        r = np.random.uniform(self.r_1, self.r_2)
        w = np.min((int(np.sqrt(s / r)), img_w - 1))
        h = np.min((int(np.sqrt(s * r)), img_h - 1))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)
        if self.random_value is None:
            c = np.random.uniform(min_value, max_value)
        else:
            c = self.random_value
        if image_dim == 2:
            output_img[top:top + h, left:left + w] = c
        else:
            if self.n_zero_channels is None:
                output_img[:, top:top + h, left:left + w] = c
            else:
                output_img[:-self.n_zero_channels,  top:top + h, left:left + w] = c
                if self.is_filled_last_channels:
                    output_img[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return output_img


class SpecAugmentNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param time_max_width: maximum time width to remove.
        :param freq_max_width: maximum freq width to remove.
        :param n_time_stripes: number of time stripes to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        n_frames = x.shape[1]
        n_freqs = x.shape[2]
        min_value = np.min(x)
        max_value = np.max(x)
        if self.time_max_width is None:
            time_max_width = int(0.15 * n_frames)
        else:
            time_max_width = self.time_max_width
        time_max_width = np.max((1, time_max_width))
        if self.freq_max_width is None:
            freq_max_width = int(0.2 * n_freqs)
        else:
            freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_time_stripes):
            dur = np.random.randint(1, time_max_width, 1)[0]
            start_idx = np.random.randint(0, n_frames - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, start_idx:start_idx + dur, :] = random_value
            else:
                new_spec[:-self.n_zero_channels, start_idx:start_idx + dur, :] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, start_idx:start_idx + dur, :] = 0.0

        for i in np.arange(self.n_freq_stripes):
            dur = np.random.randint(1, freq_max_width, 1)[0]
            start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, :, start_idx:start_idx + dur] = random_value
            else:
                new_spec[:-self.n_zero_channels, :, start_idx:start_idx + dur] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, :, start_idx:start_idx + dur] = 0.0

        return new_spec


class RandomCutoutHoleNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a few small holes in the spectrogram. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 8,
                 max_w_size: int = 8, filled_value: float = None, n_zero_channels: int = None,
                 is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum freq bands of the cutout holes.
        :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.max_h_size = np.max((max_h_size, 5))
        self.max_w_size = np.max((max_w_size, 5))
        self.filled_value = filled_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray):
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        new_spec = x.copy()
        # n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]
        n_cutout_holes = self.n_max_holes
        for ihole in np.arange(n_cutout_holes):
            # w = np.random.randint(4, self.max_w_size, 1)[0]
            # h = np.random.randint(4, self.max_h_size, 1)[0]
            w = self.max_w_size
            h = self.max_h_size
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            if self.filled_value is None:
                filled_value = np.random.uniform(min_value, max_value)
            else:
                filled_value = self.filled_value
            if self.n_zero_channels is None:
                new_spec[:, top:top + h, left:left + w] = filled_value
            else:
                new_spec[:-self.n_zero_channels, top:top + h, left:left + w] = filled_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return new_spec



def freq_mixup(data, n_swap=None):
    """Implementing the FreqMix augmentation technique.
    
    Assuming the data is of shape (channels, timesteps, freq bins)
    
    Input
        data : Spectrogram data (ch, time, freq)
        n_swap : Number of frequency bins to swap. Defaults to None, which will swap 8% of frequency bins
        
    Returns
        data : Same spectrogram with frequency bins mixed up (ch, time, freq)"""
        
    n_ch, n_time, n_freq = data.shape
    
    if n_swap is None:
        n_swap = int(n_freq * 0.08)
        
    assert n_swap/n_freq < 0.2, "Try not to swap more than 20 percent of the frequency bins at once"

    x = data.copy()
    
    f_0 = np.random.randint(0, int(n_freq - 2*n_swap))
    f_1 = np.random.randint(int(f_0 + n_swap), int(n_freq - n_swap))
    
    f_low = data[:, :, f_0:f_0+n_swap]
    f_hi  = data[:, :, f_1:f_1+n_swap]

    x[:, :, f_0:f_0+n_swap] = f_hi
    x[:, :, f_1:f_1+n_swap] = f_low
    
    return x


def time_mixup(data, target,
               low_lim = 10, upp_lim = 60):
    """Function that implements mixup in the time domain.
    Assumes that the data is in the shape of (channels, timesteps, frequencies)
    Target labels in the shape of (timesteps, ...)

    Input
        data : Array of 2 data values
        target : Array of 2 corresponding target values
        low_lim : Lower limit of the mixup portion in percentage
        upp_lim : Upper limit of the mixup portion in percentage

    Returns
        mix_data_1, mix_data_2 : Mixed up data values
        mix_target_1, mix_target_2 : Mixed up corresponding target values"""
        
    x = np.copy(data)
    y = np.copy(target)

    # Taking the i, i+1 data and target samples
    d_0 = x[0]
    d_1 = x[1]
    d_time = d_0.shape[1] # Data timesteps

    t_0 = y[0]
    t_1 = y[1]
    t_time = t_0.shape[0] # Label timesteps

    # Getting the feature downsample rate
    time_downsample = int(d_time/t_time)

    # Generate a random float value of [0.10, 0.60)
    lam = np.random.randint(low_lim,upp_lim) / 100

    # Determining the index value for the data, target timesteps
    t_index = int(np.floor(lam * t_time))
    d_index = t_index * time_downsample

    # Getting the front and back data and target segments
    d_01 = d_0[:, :d_index, :]
    d_02 = d_0[:, d_index:, :]

    d_11 = d_1[:, :d_index, :]
    d_12 = d_1[:, d_index:, :]

    t_01 = t_0[:t_index, :, :, :]
    t_02 = t_0[t_index:, :, :, :]

    t_11 = t_1[:t_index, :, :, :]
    t_12 = t_1[t_index:, :, :, :]

    # Now we combine the segmented parts
    mix_data_1 = np.concatenate((d_01, d_12), axis=1)
    mix_target_1 = np.concatenate((t_01, t_12), axis=0)

    mix_data_2 = np.concatenate((d_11, d_02), axis=1)
    mix_target_2 = np.concatenate((t_11, t_02), axis=0)

    return mix_data_1, mix_data_2, mix_target_1, mix_target_2

def tf_mixup(data, target, use_freq=False, freq_p = 0.5, n_freq_mix = None,
             use_time = False, time_p = 0.5, t_low=10, t_hi=60):
    """
    Implements the Time-Frequency Mixup data augmentation technique. 
    
    Inputs
        data : Array of 2 data values
        target : Array of 2 corresponding target values
        use_freq (boolean) : True if use Freq Mixup
        freq_p (float) : Probability of using frequency mixup
        n_freq_mix (int) : How many frequency bins to mixup. None if wish to default to mix only 8% of frequency bins
        use_time (boolean) : True if use Time Mixup
        low_lim (int) : Lower limit of the mixup portion in percentage
        upp_lim (int) : Upper limit of the mixup portion in percentage

    Returns
        mix_data_1, mix_data_2 : Mixed up data values
        mix_target_1, mix_target_2 : Mixed up corresponding target values
    """

    x = np.copy(data)
    y = np.copy(target)

    if use_time:
        if np.random.rand() < time_p:
            d1, d2, t1, t2 = time_mixup(x, y,
                                        low_lim=t_low, upp_lim=t_hi)
            data[0] = d1
            data[1] = d2
            target[0] = t1
            target[1] = t2

    if use_freq:
        for idx in range(len(data)):
            if np.random.rand() < freq_p:
                data[idx] = freq_mixup(data[idx], n_swap=n_freq_mix)
    
    return data[0], data[1], target[0], target[1]
