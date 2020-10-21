"""
Low-complexity TCL controller
Python 3 with numba enhancements
Version 11 September 2020
Simon Tindemans, s.h.tindemans@tudelft.nl

This code implements the control algorithm described in the paper 
"Low-complexity algorithm for decentralized aggregate load control of thermostatic loads",
to appear in IEEE Transactions on Industry Applications. 
Please reference this paper if you use this code.

The algorithm is a functionally equivalent but cleaner version of the algorithm in:
Simon Tindemans and Goran Strbac,
"Low-complexity control algorithm for decentralised demand response using thermostatic loads",
2019 IEEE Conference on Environment and Electrical Engineering (EEEIC 2019), Genoa (Italy).
Paper DOI: 10.1109/EEEIC.2019.8783359
Code DOI: 10.24433/CO.6765929.v1
"""
# SPDX-License-Identifier: MIT

import numpy as np
import numba as nb


@nb.experimental.jitclass([
    ('Toff', nb.float64),
    ('Ton', nb.float64),
    ('Tmax', nb.float64),
    ('Tmin', nb.float64),
    ('alpha', nb.float64),
    ('pi0', nb.float64),
    ('Tavg0', nb.float64),
    ('width', nb.float64)
])
class Model:
    """Helper class to store first order fridge model parameters"""

    def __init__(self, Toff, Ton, Tmax, Tmin, alpha, width):
        self.Toff = Toff
        self.Ton = Ton
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.alpha = alpha
        self.width = width

        self.update_ss_parameters()

    def update_ss_parameters(self):
        self.pi0 = np.log((self.Tmax-self.Ton)/(self.Tmin-self.Ton))/np.log(((self.Tmax-self.Ton)*(self.Tmin-self.Toff))/((self.Tmin-self.Ton)*(self.Tmax-self.Toff)))
        self.Tavg0 = self.Toff - (self.Toff-self.Ton)*self.pi0
        return

@nb.experimental.jitclass([
    ('state', nb.int64),
    ('snapshot_time', nb.float64),
    ('reference_power', nb.float64),
    ('z', nb.float64),
    ('rate_on_off', nb.float64),
    ('rate_off_on', nb.float64)
])
class TCLcontrolState:
    """Helper class to store TCLcontrol state information between invocations"""

    def __init__(self, state, snapshot_time, reference_power, z, rate_on_off, rate_off_on):
        assert(state == int(0) or state == int(1))
        self.state = state                      # int: 1 (on) or 0 (off)
        self.snapshot_time = snapshot_time      # time (seconds) of function invocation: t_i
        self.reference_power = reference_power  # Pi(t) attained at t=t_i
        self.z = z                              # z(t_i)
        self.rate_on_off = rate_on_off          # on->off rate for t>t_i (is used at next invocation)
        self.rate_off_on = rate_off_on          # off->on rate for t>t_i (is used at next invocation)


@nb.njit
def random_state(model, current_time=0.0):
    # if no previous state is provided, initialise in the steady state.

    state = 1 if (np.random.random() < model.pi0) else 0

    if state == 1:
        temperature = model.Ton + (model.Tmin - model.Ton) * \
                      (((model.Tmax - model.Ton)/(model.Tmin - model.Ton))**np.random.random())
    else:
        temperature = model.Toff + (model.Tmin - model.Toff) * \
                      (((model.Toff - model.Tmax)/(model.Toff - model.Tmin))**np.random.random())

    return temperature, TCLcontrolState(
        state,            # state           # random on/off initialisation
        current_time,     # snapshot_time   # delta-time = 0, so no rate-induced switching
        1.0,              # reference_power # assume Pi(t<t_i)=1
        0.0,              # z               # assume steady state initialisation
        0.0,              # rate_on_off     # steady state -> r=0
        0.0               # rate_off_on     # steady state -> r=0
    )


@nb.njit
def update_state(requested_power, current_temperature, current_time, model, previous_state):
    """
    Update function that implements the Distribution Referred Controller.

    :param requested_power: desired value of Pi(t)
    :param current_temperature: Measured control temperature
    :param current_time: Time in seconds
    :param model: Model object that describes the fridge
    :param previous_state: TCLcontrolState object that represents the saved state from the previous invocation.
    :return: tuple of (binary compressor state, TCLcontrolState)

    Notes:
    - the returned TCLcontrolState object should be supplied as previous_state in the next invocation of the function
    """

    # COMPUTE Z UPDATE

    time_delta = current_time - previous_state.snapshot_time
    decay_factor = np.exp(-model.alpha * time_delta)
    z = previous_state.z * decay_factor + (previous_state.reference_power - 1.0)*(1.0 - decay_factor)

    # compute reference temperature values R_i^- and R_-^+ in left/right limits ('pre'/'post' with respect to t_i)
    # note that limits of zeta(t), beta(t) and s(t) are not explicitly computed, because these follow from the
    # input (R_i^(-,+)) and (in the case of beta(t) from Pi
    ref_temp_pre = model.Tmax if (previous_state.z <= 0.0) else model.Tmin
    ref_temp_post = model.Tmax if (z <= 0.0) else model.Tmin

    # DEFINE CONTROL AND SCALING RELATIONS

    # This is all done in functional form; computations are postponed to the next section.
    # Explicit dependencies on R and Pi are maintained, because these values are discontinuous at the current
    # time of evaluation.
    def zeta(R):
        return (model.Tavg0 - R)/(model.Toff - model.Tavg0)
    def beta(Pi, R):
        return (Pi - 1.0 - z)/(z - zeta(R))
    def scale(R):
        return 1.0 - z/zeta(R)

    # define boundary temperatures
    def T_low(ref_temp):
        return ref_temp - (ref_temp - model.Tmin)*scale(ref_temp)
    def T_high(ref_temp):
        return ref_temp - (ref_temp - model.Tmax)*scale(ref_temp)

    # define switching rates
    def P(R):
        return (R - model.Toff) * scale(R) + (current_temperature - R)
    def Q(R):
        return (R - model.Ton) * scale(R) + (current_temperature - R)
    def X(Pi, R):
        return (current_temperature - model.Toff) + (current_temperature - R)*beta(Pi, R)
    def Y(Pi, R):
        return (current_temperature - model.Ton) + (current_temperature - R)*beta(Pi, R)
    def Xi(Pi, R):
        return model.alpha*model.alpha*(X(Pi,R)*Y(Pi,R)/(P(R)*Q(R))*(P(R)+Q(R)) - (1+beta(Pi,R))*(X(Pi,R) + Y(Pi,R)))

    # APPLY ENERGY AND POWER LIMITS

    reference_power = requested_power

    # apply energy limits
    if z <= model.width * zeta(model.Tmax):     # lower energy limit violated
        reference_power = np.maximum(reference_power, 1 + model.width * zeta(model.Tmax))
    elif z >= model.width * zeta(model.Tmin):   # upper energy limit violated
        reference_power = np.minimum(reference_power, 1 + model.width * zeta(model.Tmin))

    # apply power limits (first lower, then upper)
    reference_power = np.maximum(reference_power,
                                 1.0 + zeta(ref_temp_post)*(model.Tmin + model.Tmax - model.Toff - ref_temp_post)/
                                 (model.Tmin + model.Tmax - 2*ref_temp_post))
    reference_power = np.minimum(reference_power,
                                 1.0 + zeta(ref_temp_post)*(model.Tmin + model.Tmax - model.Ton - ref_temp_post)/
                                 (model.Tmin + model.Tmax - 2*ref_temp_post))

    # COMPUTATION OF PROBABILITIES

    # precompute X_i^- (X_pre), X_i^- (X_post), Y_i^- (Y_pre), Y_i^- (Y_post)
    X_pre = X(previous_state.reference_power, ref_temp_pre)
    X_post = X(reference_power, ref_temp_post)
    Y_pre = Y(previous_state.reference_power, ref_temp_pre)
    Y_post = Y(reference_power, ref_temp_post)

    # compute on->off rates, in left and right limits
    rate_on_off_pre = max(0, -Xi(previous_state.reference_power, ref_temp_pre) / (model.alpha*X_pre))
    rate_on_off_post = max(0, -Xi(reference_power, ref_temp_post) / (model.alpha*X_post))

    # compute off->on rates, in left and right limits
    rate_off_on_pre = max(0, -Xi(previous_state.reference_power, ref_temp_pre) / (model.alpha*Y_pre))
    rate_off_on_post = max(0, -Xi(reference_power, ref_temp_post) / (model.alpha*Y_post))

    # compute stochastic switching probabilities
    prob_off_on_stoch = 0.5*time_delta*(rate_off_on_pre + previous_state.rate_off_on)
    prob_on_off_stoch = 0.5*time_delta*(rate_on_off_pre + previous_state.rate_on_off)

    # compute instantaneous switching probabilities
    prob_off_on_inst = max(0, 1.0 - Y_post/Y_pre)
    prob_on_off_inst = max(0, 1.0 - X_post/X_pre)

    # IMPLEMENT SWITCHING ACTIONS

    if previous_state.state == 1:
        if current_temperature <= T_low(ref_temp_post):
            new_state = 0           # Tlow exceeded -> switching off
        elif np.random.random() < prob_on_off_stoch + prob_on_off_inst:
            new_state = 0           # switching off to shape distribution
        else:
            new_state = 1
    else:
        if current_temperature >= T_high(ref_temp_post):
            new_state = 1           # Thigh exceeded -> switching on
        elif np.random.random() < prob_off_on_stoch + prob_off_on_inst:
            new_state = 1           # switching on to shape distribution
        else:
            new_state = 0

    # Save relevant parts of the current state in a TCLcontrolState object for the next invocation
    # NOTE: explicit argument naming has been suppressed due to incompatibility with numba
    return_state = TCLcontrolState(
        new_state,          # state
        current_time,       # snapshot_time
        reference_power,    # reference_power
        z,                  # z
        rate_on_off_post,   # rate_on_off
        rate_off_on_post    # rate_off_on
    )

    return return_state
