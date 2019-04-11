#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   B2KstLL.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   11.04.2019
# =============================================================================
"""B -> K*ll angular distribution in zfit."""

from math import pi

import zfit


_PARAMS = []


class B2Kstll:
    def __init__(self, costheta_l, costheta_k, phi):
        self._obs_names = {'costheta_l': costheta_l.obs,
                           'costheta_k': costheta_k.obs,
                           'phi': phi.obs}
        self.obs = costheta_l * costheta_k * phi
        self.params = self.create_parameters()

    @staticmethod
    def create_parameters():
        params = {'FL': zfit.Parameter('FL', 0.8, 0.0, 1.0)}
        params.update({f'S{i}': zfit.Parameter(f'S{i}', 0.0, -1.0, 1.0)
                       for i in range(3,10)})
        return params


if __name__ == "__main__":
    costheta_l = zfit.Space("costhetal", limits=(-1.0, 1.0))
    costheta_k = zfit.Space("costhetaK", limits=(-1.0, 1.0))
    phi = zfit.Space("phi", limits=(-pi, pi))
    B2Kstll(costheta_l, costheta_k, phi)

# EOF