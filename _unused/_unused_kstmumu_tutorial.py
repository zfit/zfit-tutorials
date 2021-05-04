"""Tutorial skeleton for generation and fit, including simultaneous, of B->K*mumu with determination of the
significance.

@author: Jonas Eschle
@email: Jonas.Eschle@cern.ch
@date: 21.02.2020
"""

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from particle.particle import literals as lp
from phasespace import GenParticle

seed = 11  # "good" seed: 11
tf.random.set_seed(seed)
np.random.seed(seed)

# TODO: @Eduardo, maybe change this to demonstrate the particle package? Alternative ways
KSTARZ_MASS = lp.Kst_892_0.mass
KSTARZ_WIDTH = lp.Kst_892_0.width
B0_MASS = lp.B_0.mass
PION_MASS = lp.pi_minus.mass
KAON_MASS = lp.K_plus.mass
MU_MASS = lp.mu_minus.mass
JPSI_MASS = lp.Jpsi_1S.mass

# overall parameters to change
n_sig_rare = 120
n_sig_reso = 4000
n_bkg_rare = 5000
n_bkg_reso = 3000

rare_smearing = 7  # detector smearing of the particles 4-momenta (sigma of a gaussian)
reso_smearing = 7


# *************************************************
# PHASESPACE GENERATION OF RARE SIGNAL
# *************************************************

# kstar is resonant, we define here the sampling of the mass
# therefore we can use zfit PDF or TensorFlow Probability (or anything if wrappet with `tf.py_function`
def kstar_mass(min_mass, max_mass, n_events):
    min_mass = tf.cast(min_mass, tf.float64)
    max_mass = tf.cast(max_mass, tf.float64)
    kstar_width_cast = tf.cast(KSTARZ_WIDTH, tf.float64)
    kstar_mass_cast = tf.cast(KSTARZ_MASS, dtype=tf.float64)

    kstar_mass = tf.broadcast_to(kstar_mass_cast, shape=(n_events,))
    if KSTARZ_WIDTH > 0:
        kstar_mass = tfp.distributions.TruncatedNormal(loc=kstar_mass,
                                                       scale=kstar_width_cast,
                                                       low=min_mass,
                                                       high=max_mass).sample()
    return kstar_mass


kstar = GenParticle('K*0', mass=kstar_mass).set_children(GenParticle('K+', mass=KAON_MASS),
                                                         GenParticle('pi-', mass=PION_MASS))
bz = GenParticle('B0', B0_MASS).set_children(kstar,
                                             GenParticle('mu+', mass=MU_MASS),
                                             GenParticle('mu-', mass=MU_MASS)
                                             )
weights, particles = bz.generate(n_sig_rare)
weights = weights / np.average(weights)


# to make it look real (similar to what RapidSim does), we smear the particles
def smear_momenta(four_momenta, smearing=10):
    # the four_momenta are "eager_tensors", wrapped numpy arrays. We could convert them with `np.array(four_momenta)`
    return np.random.normal(loc=four_momenta, scale=smearing)


def invariant_mass(four_momenta):
    momenta_squared = four_momenta ** 2
    return np.sqrt(momenta_squared[:, 3] - np.sum((momenta_squared[:, :3]), axis=-1))


smeared_momenta = {}
daugther_particles = ['K+', 'pi-', 'mu+', 'mu-']
for particle in daugther_particles:
    smeared_momenta[particle] = smear_momenta(particles[particle], smearing=rare_smearing)

smeared_momenta['K*0'] = smeared_momenta['K+'] + smeared_momenta['pi-']
smeared_momenta['Jpsi'] = smeared_momenta['mu+'] + smeared_momenta['mu-']
smeared_momenta['B0'] = smeared_momenta['K*0'] + smeared_momenta['Jpsi']

b_mass_rare = invariant_mass(smeared_momenta['B0'])
q2 = invariant_mass(smeared_momenta['Jpsi'])

# plot the b mass with the weights. It is basically the same as without weights.
plt.figure()
plt.title("B mass generated non resonant")
plt.hist(b_mass_rare, weights=weights, alpha=0.5, bins=40, label="with weights")
plt.hist(b_mass_rare, bins=40, alpha=0.5, label="no weights")
plt.legend()

plt.figure()
plt.title("q2 generated non resonant")
plt.hist(q2, weights=weights, alpha=0.5, bins=40, label="with weights")
plt.hist(q2, bins=40, alpha=0.5, label="no weights")
plt.legend()

# *************************************************
# START FIT
# *************************************************
#
# The fit goes in three steps:
# - a fit with the exponential to the right side to have a good
#   starting value.
# - a model composed of an exponential for the combinatorial bkg and a double Crystalball for
#   the signal is built and fitted to the "rare" mode
# - the same model with a few shared parameters is built for the resonant mode to improve the shape
#   of the DoubleCB, as moslty the tails are tricky. They seem to be independent of q2, so we
#   can share them

# --------------------------------------------------
# RIGHT SIDE BAND FIT
# --------------------------------------------------

import zfit

# TODO @Eduardo: maybe adjust the range?
upper_limit = 5600
obs = zfit.Space('Bmass', (5000, upper_limit))  # for whole range
obs_bkg = zfit.Space('Bmass', (5400, upper_limit))  # to pre-fit the exponential

# Parameters are specified:  (name (unique), initial, lower, upper) whereas lower, upper are optional
lambda_rare = zfit.Parameter('lambda_rare', -0.002, -0.01, -0.0001, step_size=0.001)  # floating, also without limits
comb_bkg_rare = zfit.pdf.Exponential(lambda_rare, obs=obs)

# create some bkg data
comb_bkg_rare_sample = comb_bkg_rare.sample(n=n_bkg_rare)  # sampled within the limits of `obs`

# to improve our fit, we can prefit the rightside
right_tale_data_rare = zfit.Data.from_numpy(obs=obs_bkg, array=comb_bkg_rare_sample.value())

# set the value of lambda to smth different than we sampled from (for the fit afterwards)
lambda_rare.set_value(-0.003)

# here we temporarily set the normalization range to the right side only
with comb_bkg_rare.set_norm_range(obs_bkg):
    right_tale_loss = zfit.loss.UnbinnedNLL(comb_bkg_rare, right_tale_data_rare)
    minimizer = zfit.minimize.Minuit(verbosity=7)
    result_right_tale = minimizer.minimize(right_tale_loss)

# --------------------------------------------------
# RARE MODE FIT
# --------------------------------------------------

# now create the data for the rare fit
rare_data_np = np.concatenate([b_mass_rare, comb_bkg_rare_sample[:, 0]], axis=0)
rare_weights_np = np.concatenate([weights, np.ones_like(comb_bkg_rare_sample[:, 0])], axis=0)
# TODO @Eduardo, we could do some data preprocessing here (e.g. apply a cut to q2, which we actually
# need to have "no Jpsi" in there. This can be done with pandas and then loaded into zfit

rare_data = zfit.Data.from_numpy(obs=obs, array=rare_data_np, weights=rare_weights_np)
# we can also load data from ROOT
# right_tale_data_rare = zfit.Data.from_root(...)

# ...or to/from a pandas DataFrame. Either convert the `zfit.Data` to a pandas DF
# right_tale_data_rare_df = right_tale_data_rare.to_pandas()
# .... or create it from scratch
# right_tale_data_rare_df = pd.DataFrame(data=rare_data_np, columns=obs_bkg.obs)
# Then we can directly load it
# right_tale_data_rare = zfit.Data.from_pandas(df=right_tale_data_rare_df, obs=obs_bkg)

# TODO @eduardo: maybe remove plot, just for data visualization example?
plt.figure()
plt.title("B mass and bkg")
plt.hist(rare_data_np, weights=rare_weights_np, bins=40)

# create the model to fit

# set the normalization range of the exponential to the whole range
comb_bkg_rare.set_norm_range(obs)

# parameters for the model
mu = zfit.Parameter('mu', 5270, 5200, 5350)
sigma = zfit.Parameter('sigma', 22, 0, 100)

# we could also use a smple model here
# signal_rare = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# We will use a double crystall ball
alphal_rare = zfit.Parameter('alpha left rare', -0.3, -5, 0)
nl_rare = zfit.Parameter('n left rare', 0.3, 0, 10)
alphar_rare = zfit.Parameter('alpha right rare', 1, 0, 5)
nr_rare = zfit.Parameter('n right rare', 2.8, 0, 10)
frac_dcb_rare = zfit.Parameter('frac dcb', 0.3, 0.1, 0.9)

left_cb_rare = zfit.pdf.CrystalBall(obs=obs,
                                    mu=mu, sigma=sigma,
                                    alpha=alphal_rare, n=nl_rare,
                                    )
right_cb_rare = zfit.pdf.CrystalBall(obs=obs,
                                     mu=mu, sigma=sigma,
                                     alpha=alphar_rare, n=nr_rare,
                                     )
signal_rare = zfit.pdf.SumPDF([left_cb_rare, right_cb_rare], fracs=frac_dcb_rare)

# now create the yields and the extended pdfs
rare_sig_yield = zfit.Parameter('rare_sig_yield', n_sig_rare + 30,
                                step_size=3)  # step size: default is small, use appropriate
rare_bkg_yield = zfit.Parameter('rare_bkg_yield', n_bkg_rare - 40, step_size=1)
extended_sig_rare = signal_rare.create_extended(rare_sig_yield)
extended_bkg_rare = comb_bkg_rare.create_extended(rare_bkg_yield)
model_rare = zfit.pdf.SumPDF([extended_bkg_rare, extended_sig_rare])


# in order to plot our model (before the fit), we create a helper here
def plot_pdf_data(data, model, title, n_bins=40):
    linewidth = 2.5
    space = data.data_range
    plot_scaling = data.nevents / n_bins * space.area()
    lower, upper = space.limit1d
    x = np.linspace(lower, upper, 1000)

    plt.figure()
    plt.title(title)

    # plot the data
    data_np = data[:, 0]
    plt.hist(data_np,
             # color=color,
             bins=n_bins, histtype="stepfilled", alpha=0.1)
    plt.hist(data_np,
             # color=color,
             bins=n_bins, histtype="step")
    # plot the pdfs
    y = model.pdf(x).numpy()
    y_sig = (model.pdfs[0].pdf(x) * model.fracs[0]).numpy()  # notice the frac!
    y_bkg = (model.pdfs[1].pdf(x) * model.fracs[1]).numpy()  # notice the frac!

    plt.plot(x, y * plot_scaling, label="Sum - Model", linewidth=linewidth * 2)
    plt.plot(x, y_sig * plot_scaling, '--', label=f"{model.pdfs[0].name} - Signal", linewidth=linewidth)
    plt.plot(x, y_bkg * plot_scaling, '--', label=f"{model.pdfs[1].name} - Background", linewidth=linewidth)
    plt.xlabel(space.obs[0])
    plt.legend()


plot_pdf_data(data=rare_data, model=model_rare, title=r'before fitting: $B^0 -> K^{*} (-> K^+ \pi^-) \mu^+ \mu^-$')
plt.show()

# create a loss: the minimum of it well-defines the solution to our problem
ext_rare_nll = zfit.loss.ExtendedUnbinnedNLL(model_rare, rare_data)

result_rare = minimizer.minimize(ext_rare_nll)
# we could also specify the params explicitly; now all floating are used
plot_pdf_data(data=rare_data, model=model_rare, title=r'after rare fitting: $B^0 -> K^{*} (-> K^+ \pi^-) \mu^+ \mu^-$')
plt.show()
# all the parameters are set to the minimum of the fit. To store the fit information, we can
# use the FitResult that is returned. It contains e.g. information about the parameter
pprint(result_rare.params)

# ...and more:
print(f"The fit converged: {result_rare.converged}, the minimum is {result_rare.fmin}")

# the following are error estimations. They add their result into the `params` attribute.
# Unfortunately, hesse is currently not yet supported with weights.
# result_rare.hesse()  # error calculation using the inverse hessian approximation
# result_rare.error()  # error calculation using minos, this takes all parameters (expensive)
result_rare.error([rare_sig_yield, mu])  # just for specific parameters
pprint(result_rare.params)

# the params can be accesssed using the parameter objects
mu_rare_fit = result_rare.params[mu]
# they contain information about the result such as value etc
print(f"Mu value of rare fit: {mu_rare_fit['value']} "
      f"+ {mu_rare_fit['minuit_minos']['upper']} "
      f"- {mu_rare_fit['minuit_minos']['upper']}"
      # f" (symmetric Hesse error: {mu_rare_fit['minuit_hesse']['error']})"
      )

# *************************************************
# PHASESPACE GENERATION OF RESONANT SIGNAL
# *************************************************
# Since the previous fit, e.g. the tail, was not so great due to the lack of statistics, we can
# also fit the resonant mode and share certain parameters

# create the resonant decay
bz = GenParticle('B0', B0_MASS).set_children(kstar,
                                             GenParticle('Jpsi', mass=JPSI_MASS).set_children(
                                                 GenParticle('mu+', mass=MU_MASS),
                                                 GenParticle('mu-', mass=MU_MASS)
                                             ))

weights_reso, particles_reso = bz.generate(n_sig_reso)
weights_reso /= np.average(weights_reso)

smeared_momenta_reso = {}
daugther_particles_reso = ['K+', 'pi-', 'mu+', 'mu-']
for particle in daugther_particles_reso:
    smeared_momenta_reso[particle] = smear_momenta(particles_reso[particle], smearing=reso_smearing)

smeared_momenta_reso['K*0'] = smeared_momenta_reso['K+'] + smeared_momenta_reso['pi-']
smeared_momenta_reso['Jpsi'] = smeared_momenta_reso['mu+'] + smeared_momenta_reso['mu-']
smeared_momenta_reso['B0'] = smeared_momenta_reso['K*0'] + smeared_momenta_reso['Jpsi']

b_mass_reso = invariant_mass(smeared_momenta_reso['B0'])
q2_reso = invariant_mass(smeared_momenta_reso['Jpsi'])

# plot the q2
plt.figure()
plt.title("q2 generated resonant")
plt.hist(q2_reso, weights=weights_reso, alpha=0.5, bins=40, label="with weights")
plt.hist(q2_reso, bins=40, alpha=0.5, label="no weights")
plt.legend()

# ------------------------------------------------
# COMBINATORIAL BACKGROUND RESO
# ------------------------------------------------

lambda_reso = zfit.Parameter('lambda_reso', -0.002, -0.01, 0.0001)  # floating, also without limits
comb_bkg_reso_pdf = zfit.pdf.Exponential(lambda_reso, obs=obs)

# create some more bkg data
comb_bkg_reso_sample = comb_bkg_reso_pdf.sample(n=n_bkg_reso)  # sampled within the limits of `obs`

# set the value of lambda to smth different then we sampled from (for the fit afterwards)
lambda_reso.set_value(-0.01)

reso_data_np = np.concatenate([b_mass_reso, comb_bkg_reso_sample[:, 0]], axis=0)
reso_weights_np = np.concatenate([weights_reso, np.ones_like(comb_bkg_reso_sample[:, 0])], axis=0)

# TODO @eduardo: maybe remove plot, just for data visualization example?
plt.figure()
plt.title("resonant data")
plt.hist(reso_data_np, weights=reso_weights_np, bins=40)

reso_data = zfit.Data.from_numpy(obs=obs, array=reso_data_np)

# ------------------------------------------------
# RESONANT MODEL
# ------------------------------------------------

# create the model to fit

# we can share parameters directly or create composed parameters. Here we have a
# parameter that scales the sigma from the rare fit

sigma_scaling = zfit.Parameter('sigma_scaling', 0.9, 0.1, 10)


def sigma_scaled_fn():
    return sigma * sigma_scaling  # this can be an arbitrary function


sigma_scaled = zfit.ComposedParameter('sigma scaled', sigma_scaled_fn,
                                      dependents=sigma  # the objects used inside the func
                                      )

# we could also make the free parameters, not shared
# alphal_reso = zfit.Parameter('alpha left reso', -0.7, -5, 0)
# nl_reso = zfit.Parameter('n left reso', 0.4, 0, 10)
# alphar_reso = zfit.Parameter('alpha right reso', 1, 0, 5)
# nr_reso = zfit.Parameter('n right reso', 1.8, 0, 10)

alphal_reso = alphal_rare
nl_reso = nl_rare
alphar_reso = alphal_rare
nr_reso = nr_rare

# frac_dcb_reso = zfit.Parameter('frac dcb_reso', 0.5, 0.01, 0.99)
frac_dcb_reso = frac_dcb_rare
left_cb_reso = zfit.pdf.CrystalBall(obs=obs,
                                    mu=mu, sigma=sigma_scaled,
                                    alpha=alphal_reso, n=nl_reso,
                                    )
right_cb_reso = zfit.pdf.CrystalBall(obs=obs,
                                     mu=mu, sigma=sigma_scaled,
                                     alpha=alphar_reso, n=nr_reso,
                                     )
signal_reso = zfit.pdf.SumPDF([left_cb_reso, right_cb_reso], fracs=frac_dcb_reso)

# or we can use a simpler shape
# signal_reso = zfit.pdf.Gauss(mu=mu,  # using the same mu as above means it's shared
#                              sigma=sigma_scaled, obs=obs)

reso_sig_yield = zfit.Parameter('reso_sig_yield', n_sig_reso - 100, 0, n_sig_reso * 3,
                                step_size=1)  # step size: default is small, use appropriate
reso_bkg_yield = zfit.Parameter('reso_bkg_yield', n_bkg_reso + 70, 0, n_bkg_reso * 3, step_size=1)
extended_sig_reso = signal_reso.create_extended(reso_sig_yield)
extended_bkg_reso = comb_bkg_reso_pdf.create_extended(reso_bkg_yield)
model_reso = zfit.pdf.SumPDF([extended_bkg_reso, extended_sig_reso])

# ------------------------------------------------
# SIMULTANEOUS LOSS
# ------------------------------------------------

# we could also add a constraint to the loss
# constraint = zfit.constraint.GaussianConstraint(mu, observation=5279, uncertainty=50)
# ext_reso_nll = zfit.loss.ExtendedUnbinnedNLL(model_reso, reso_data, constraints=constraint)
ext_reso_nll = zfit.loss.ExtendedUnbinnedNLL(model_reso, reso_data)

# to create a simultaneous loss, we simply add them
simultaneous_loss = ext_reso_nll + ext_rare_nll

result_simult = minimizer.minimize(simultaneous_loss)

plot_pdf_data(data=rare_data, model=model_rare, title=r'$B^0 -> K^{*} (-> K^+ \pi^-) \mu^+ \mu^-$')
plot_pdf_data(data=reso_data, model=model_reso, title=r'$B^0 -> K^{*} (-> K^+ \pi^-) J/\psi (-> \mu^+ \mu^-)$')
plt.show()

# Hesse is not yet supported with weights
# result_simult.hesse()  # error calculation using hesse
errors = result_simult.error([mu, rare_sig_yield, reso_sig_yield])  # error calculation using minos, just for a few
# parameters as it is quite expensive
pprint(errors)

# ------------------------------------------------
# DETERMINING THE SIGNIFICANCE
# ------------------------------------------------

# using hepstats for the limits
from hepstats.hypotests import Discovery
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests.parameters import POI

calculator = AsymptoticCalculator(simultaneous_loss, minimizer)
poinull = POI(rare_sig_yield, 0)
discovery_test = Discovery(calculator, poinull)
pnull, significance = discovery_test.result()
print(f'pnull: {pnull} with significance {significance}')
