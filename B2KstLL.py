#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   B2KstLL.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   11.04.2019
# =============================================================================
"""B -> K*ll angular distribution in zfit."""

from math import pi

from typing import Type

import tensorflow as tf
import numpy as np
import pandas as pd

import zfit

ztf = zfit.ztf
ztyping = zfit.util.ztyping
ztypes = zfit.settings.ztypes


# The PDFs

class P5pPDF(zfit.pdf.BasePDF):
    """P5prime observable from Bd -> Kst ll (l=e,mu)"""

    def __init__(self, FL: ztyping.ParamTypeInput, AT2: ztyping.ParamTypeInput,
                 P5p: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput,
                 name: str = "P5pPDF", dtype: Type = ztypes.float):

        """Angular distribution obtained from a fold tecnhique,
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
        parameters = {'FL': FL,
                      'AT2': AT2,
                      'P5p': P5p}
        super().__init__(obs=obs, dtype=dtype, name=name, parameters=parameters)

    def _unnormalized_pdf(self, x):
        FL = self.parameters['FL']
        AT2 = self.parameters['AT2']
        P5p = self.parameters['P5p']
        costheta_k, costheta_l, phi = ztf.unstack_x(x)

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


def fold_S5(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_S5'] = data[costheta_k]
    data[f'{phi}_S5'] = np.where(data[phi] >= pi,
                                 data[phi] - (2.0 * pi),
                                 data[phi])
    data[f'{phi}_S5'] = np.where(data[f'{phi}_S5'] < 0,
                                 -data[f'{phi}_S5'],
                                 data[f'{phi}_S5'])
    data[f'{costheta_l}_S5'] = np.where(theta_l > 0.5*pi,
                                        np.cos(pi - theta_l),
                                        data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_S5',
                                           f'{costheta_k}_S5',
                                           f'{phi}_S5'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_S5': costheta_l,
                                                       f'{costheta_k}_S5': costheta_k,
                                                       f'{phi}_S5': phi}))


# A bit of handling

class B2Kstll:
    FOLDS = {'P5p': (P5pPDF, ['FL', 'AT2', 'P5p'], fold_S5)}

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
        pdf = pdf_class(self.obs, **params)
        return pdf

    def fold_dataset(self, name, dataset):
        *_, data_transform = self.FOLDS[name]
        return data_transform(dataset, self.obs.obs)


if __name__ == "__main__":
    costheta_l = zfit.Space("costhetal", limits=(-1.0, 1.0))
    costheta_k = zfit.Space("costhetaK", limits=(-1.0, 1.0))
    phi = zfit.Space("phi", limits=(-pi, pi))
    B2Kstll(costheta_l, costheta_k, phi)

# EOFs