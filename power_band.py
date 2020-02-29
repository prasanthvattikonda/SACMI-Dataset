import numpy as np

def power_band(signal, f1=2500, f2=5000, sampling_rate=25000):
    x = signal
    y = abs(np.fft.fft(x, axis=0))
    N = len(y)
    fr = np.arange(N) * sampling_rate / N
    # Avoid aliasing
    fr = fr[range(int(N / 2))]
    y = y[range(int(N / 2)), :]
    pwr_band = sum(y[(fr >= f1) & (fr <= f2)]) / (f2 - f1)
    return(np.asarray(pwr_band))
