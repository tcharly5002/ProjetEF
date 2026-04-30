import numpy as np


def get_fluid_temperature(t, T_init, alpha, t_start_out):
    """
    Calcule T_f_in et T_f_out en fonction du temps
    """

    T_in = T_init - alpha * np.sqrt(t) #température à l'entrée du tube

    if t <= t_start_out:
        T_out = T_init
    else:
        T_out = T_init - alpha * np.sqrt(t - t_start_out) #température à la sortie du tube


    return T_in, T_out