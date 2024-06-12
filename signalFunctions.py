import numpy as np
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
