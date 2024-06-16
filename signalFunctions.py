import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from scipy.interpolate import CubicSpline


def cubic_envelope(array: np.array, add_plateau: bool = False) -> tuple[np.array, np.array]:
    """
    Создание верхней и нижней огибающих сигнала
    :param array: 1D np.array, сигнал
    :param add_plateau: boolean, вычисление граничных точек плато, не являющихся экстремумами.
    :return: tuple[np.array, np.array], списки точек верхней и нижней огибающих сигнала
    """
    q_u = np.zeros(array.shape)
    q_l = np.zeros(array.shape)

    # Первое значение огибающих, назначаем верхней и нижней одну точку начала
    u_x = [0, ]
    u_y = [array[0], ]

    l_x = [0, ]
    l_y = [array[0], ]

    # Поиск экстремумов и добавление их в список точек огибающей
    for k in range(1, len(array) - 1):
        if (array[k] > array[k - 1]) and (array[k] > array[k + 1]):
            u_x.append(k)
            u_y.append(array[k])

        if (array[k] < array[k - 1]) and (array[k] < array[k + 1]):
            l_x.append(k)
            l_y.append(array[k])

        # Вычисление знаков идет только при addPlateau == True, ^ - XOR
        if add_plateau:
            if (((array[k] > array[k - 1]) and (array[k] == array[k + 1]))
                    ^
                    ((array[k] > array[k + 1]) and (array[k] == array[k - 1]))):
                u_x.append(k)
                u_y.append(array[k])

            if (((array[k] < array[k - 1]) and (array[k] == array[k + 1]))
                    ^
                    ((array[k] < array[k + 1]) and (array[k] == array[k - 1]))):
                l_x.append(k)
                l_y.append(array[k])

    # Последнее значение огибающих, назначаем верхней и нижней одну точку конца
    u_x.append(len(array) - 1)
    u_y.append(array[-1])

    l_x.append(len(array) - 1)
    l_y.append(array[-1])

    # Создание кубических сплайнов, огибающих
    u_func = CubicSpline(u_x, u_y)
    l_func = CubicSpline(l_x, l_y)

    # Заполнение массивов огибающих значениями кубических сплайнов
    for k in range(0, len(array)):
        q_u[k] = u_func(k)
        q_l[k] = l_func(k)

    return q_l, q_u


def calculate_speed(array: np.array) -> np.array:
    """
    Вычисление скорости по ускорению сигнала интегрированием по методу трапеции
    :param array: 1D np.array, значения ускорения сигнала
    :return: 1D np.array, значения скорости сигнала
    """
    res = np.zeros(len(array))

    for i in range(len(array) - 1):
        res[i] = np.trapz(array[i:i + 2])

    # Повторение последнего вычисленного значения скорости
    res[-1] = res[-2]

    return res


def get_spectrogram(signal: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    Получение спектрограммы по сигналу с использованием окна Ханна
    :param signal: 2D torch.Tensor, значения одноканального сигнала
    :param n_fft: Ширина окна преобразования Фурье, количество анализируемых частот
    :return: Спектрограмма сигнала
    """
    window = torch.hann_window(window_length=n_fft)

    spectrum = torch.stft(signal, n_fft=n_fft, return_complex=True, window=window)
    spectrogram = spectrum.abs().pow(2)

    return spectrogram[0]


def get_spectrum(signal: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    Получение спектра сигнала с использованием окна Ханна
    :param signal: 2D torch.Tensor, значения одноканального сигнала
    :param n_fft: Ширина окна преобразования Фурье, количество анализируемых частот
    :return: Спектр сигнала
    """
    window = torch.hann_window(window_length=n_fft)

    spectrum = torch.stft(signal, n_fft=n_fft, return_complex=True, window=window).squeeze()
    spectrum = spectrum.abs().pow(2).sum(dim=1)

    return spectrum

def plot_spectrogram(spectrogram: torch.Tensor, isLog: bool = True):
    """
    Построение графика спектрограммы
    :param spectrum: спектрограмма размерности [N, T],
            где N - количество частот, T - количество окон
    :param isLog: Логарифмирование значеий для улучшения визуализации
    """
    plt.figure(figsize=(20, 5))
    plt.ylabel('Frequency (Hz)', size=20)
    plt.xlabel('Time (sec)', size=20)
    if isLog:
        plt.pcolormesh(spectrogram.log())
    else:
        plt.pcolormesh(spectrogram)
    plt.show()


def plot_spectrum(spectrum: np.array):
    """
    Построение графика спектра сигнала
    :param spectrum: Значения спектра сигнала
    """
    plt.figure(figsize=(20, 5))
    plt.plot(spectrum)
    plt.grid()
    plt.xlabel('Frequency (Hz)', size=20)
    plt.ylabel('Magnitude$^2$ / Power', size=20)
    plt.show()


# wav, sr = torchaudio.load("audio/IDS_Подшипник качения 3311A-2Z C3MT33 (SKF)  Измерение от 24.01.2024 9 11 42_1В.wav")

# speed = calculate_speed(wav[0])
# lower, upper = cubic_envelope(wav[0], True)

# spec = get_spectrum(wav, n_fft=512)
# spec = get_spectrum(torch.from_numpy(speed), n_fft=512)
# spec = get_spectrum(torch.from_numpy(upper), n_fft=512)

# plot_spectrum(spec)
