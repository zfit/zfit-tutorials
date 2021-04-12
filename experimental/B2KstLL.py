#!/usr/bin/env python3
# =============================================================================
# @file   B2KstLL.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   11.04.2019
# =============================================================================
"""B -> K*ll angular distribution in zfit."""

from math import pi
from typing import Type

import numpy as np
import pandas as pd
import tensorflow as tf
import zfit
from zfit import z

ztyping = zfit.util.ztyping
ztypes = zfit.settings.ztypes

# The PDFs

class P4pPDF(zfit.pdf.ZPDF):
    """P4prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [0, pi]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P4p (`zfit.Parameter`): Defined as S4/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):
    """
    _PARAMS = ['FL', 'AT2', 'P4p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.parameters['FL']
        AT2 = self.parameters['AT2']
        P4p = self.parameters['P4p']
        costheta_k, costheta_l, phi = z.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
            FL * costheta_k * costheta_k + \
            (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
            -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
            (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
            tf.sqrt(FL * (1 - FL)) * P4p * sin2theta_k * sin2theta_l * tf.cos(phi)

        return pdf


class P5pPDF(zfit.pdf.ZPDF):
    """P5prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [0, pi]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P5p (`zfit.Parameter`): Defined as S5/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):
    """
    _PARAMS = ['FL', 'AT2', 'P5p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.parameters['FL']
        AT2 = self.parameters['AT2']
        P5p = self.parameters['P5p']
        costheta_k, costheta_l, phi = z.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
            FL * costheta_k * costheta_k + \
            (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
            -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
            (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
            tf.sqrt(FL * (1 - FL)) * P5p * sin2theta_k * sintheta_l * tf.cos(phi)

        return pdf

class P6pPDF(zfit.pdf.ZPDF):
    """P6prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [-pi/2, pi/2]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P6p (`zfit.Parameter`): Defined as S5/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):
    """
    _PARAMS = ['FL', 'AT2', 'P6p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.parameters['FL']
        AT2 = self.parameters['AT2']
        P6p = self.parameters['P6p']
        costheta_k, costheta_l, phi = z.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
            FL * costheta_k * costheta_k + \
            (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
            -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
            (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
            tf.sqrt(FL * (1 - FL)) * P6p * sin2theta_k * sintheta_l * tf.sin(phi)

        return pdf

class P8pPDF(zfit.pdf.ZPDF):
    """P8prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [-pi/2, pi/2]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P8p (`zfit.Parameter`): Defined as S5/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):
    """
    _PARAMS = ['FL', 'AT2', 'P8p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.parameters['FL']
        AT2 = self.parameters['AT2']
        P8p = self.parameters['P8p']
        costheta_k, costheta_l, phi = z.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
            FL * costheta_k * costheta_k + \
            (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
            -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
            (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
            tf.sqrt(FL * (1 - FL)) * P8p * sin2theta_k * sin2theta_l * tf.sin(phi)

        return pdf

# Folding data

def fold_P4p(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P4p'] = data[costheta_k]
    data[f'{phi}_P4p'] = np.where(data[phi] < 0,
                                 -data[phi],
                                 data[phi])
    data[f'{phi}_P4p'] = np.where(theta_l > 0.5*pi,
                                 pi - data[f'{phi}_P4p'],
                                 data[f'{phi}_P4p'])
    data[f'{costheta_l}_P4p'] = np.where(theta_l > 0.5*pi,
                                        np.cos(pi - theta_l),
                                        data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P4p',
                                           f'{costheta_k}_P4p',
                                           f'{phi}_P4p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P4p': costheta_l,
                                                       f'{costheta_k}_P4p': costheta_k,
                                                       f'{phi}_P4p': phi}))

def fold_P5p(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P5p'] = data[costheta_k]
    data[f'{phi}_P5p'] = np.where(data[f'{phi}_P5p'] < 0,
                                 -data[f'{phi}_P5p'],
                                 data[f'{phi}_P5p'])
    data[f'{costheta_l}_P5p'] = np.where(theta_l > 0.5*pi,
                                        np.cos(pi - theta_l),
                                        data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P5p',
                                           f'{costheta_k}_P5p',
                                           f'{phi}_P5p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P5p': costheta_l,
                                                       f'{costheta_k}_P5p': costheta_k,
                                                       f'{phi}_P5p': phi}))

def fold_P6p(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P6p'] = data[costheta_k]
    data[f'{phi}_P6p'] = np.where(data[phi] > 0.5*pi,
                                 pi - data[phi],
                                 data[phi])
    data[f'{phi}_P6p'] = np.where(data[f'{phi}_P6p'] < - 0.5*pi,
                                 - pi - data[f'{phi}_P6p'],
                                 data[f'{phi}_P6p'])
    data[f'{costheta_l}_P6p'] = np.where(theta_l > 0.5*pi,
                                        np.cos(pi - theta_l),
                                        data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P6p',
                                           f'{costheta_k}_P6p',
                                           f'{phi}_P6p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P6p': costheta_l,
                                                       f'{costheta_k}_P6p': costheta_k,
                                                       f'{phi}_P6p': phi}))

def fold_P8p(data, costheta_k, costheta_l, phi):
    theta_k = np.acos(data[costheta_k])
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P8p'] = np.where(theta_l > 0.5*pi,
                                np.cos(pi - theta_k),
                                data[costheta_k])

    data[f'{phi}_P8p'] = np.where(data[phi] > 0.5*pi,
                                 pi - data[phi],
                                 data[phi])
    data[f'{phi}_P8p'] = np.where(data[f'{phi}_P8p'] < - 0.5*pi,
                                 - pi - data[f'{phi}_P8p'],
                                 data[f'{phi}_P8p'])
    data[f'{costheta_l}_P8p'] = np.where(theta_l > 0.5*pi,
                                        np.cos(pi - theta_l),
                                        data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P8p',
                                           f'{costheta_k}_P8p',
                                           f'{phi}_P8p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P8p': costheta_l,
                                                       f'{costheta_k}_P8p': costheta_k,
                                                       f'{phi}_P8p': phi}))

# A bit of handling

class B2Kstll:
    FOLDS = {'P4p': (P4pPDF, ['FL', 'AT2', 'P4p'], fold_P4p),
             'P5p': (P5pPDF, ['FL', 'AT2', 'P5p'], fold_P5p),
             'P6p': (P6pPDF, ['FL', 'AT2', 'P6p'], fold_P6p),
             'P8p': (P8pPDF, ['FL', 'AT2', 'P8p'], fold_P8p)}

    def __init__(self, costheta_l, costheta_k, phi):
        self._obs_names = {'costheta_l': costheta_l.obs,
                           'costheta_k': costheta_k.obs,
                           'phi': phi.obs}
        self.obs = costheta_l * costheta_k * phi
        self.params = {}

    def get_folded_pdf(self, name):
        pdf_class, param_names, _ = self.FOLDS[name]

        def get_params(param_list):
            out = {}
            for param in param_list:
                if param not in self.params:
                    config = [0.8, 0, 1] if param == 'FL' else [0.0, -1, 1]
                    self.params.update({param: zfit.Parameter(param, *config)})
                out[param] = self.params[param]
            return out

        # Make sure params exist
        params = get_params(param_names)
        pdf = pdf_class(obs=self.obs, **params)
        return pdf

    def fold_dataset(self, name, dataset):
        *_, data_transform = self.FOLDS[name]
        return data_transform(dataset, self.obs.obs)


if __name__ == "__main__":
    costheta_l = zfit.Space("costhetal", limits=(-1.0, 1.0))
    costheta_k = zfit.Space("costhetaK", limits=(-1.0, 1.0))
    phi = zfit.Space("phi", limits=(-pi, pi))
    decay = B2Kstll(costheta_l, costheta_k, phi)
    pdf = decay.get_folded_pdf('P5p')

# EOFs
