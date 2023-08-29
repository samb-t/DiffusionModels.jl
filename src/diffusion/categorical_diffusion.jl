# Notes:
# A continuous time version of this needs to have a change function
# which is a function of time, and decides when a transition should
# take place

# Have a read of "Score-Based Continuous-Time Discrete Diffusion Models"
# https://arxiv.org/abs/2211.16750

"""
    Absorbing Diffusion Models
"""
struct CategoricalDiffusion <: CategoricalDiffusion
    ...
end

# For both discrete and continuous time, decide approximately how many 
# changes have taken place by this time, and make that many changes
function marginal()
    ...
end

# Discrete and continuous time are similar. In this time period, decide
# the probability of a per-dimension change occuring, and make mask out
# Discrete time:
#   - What is the probability of masking at this step
# Continuous time:
#   - What is the probability that a token was masked in the presceeding
#     time period.
function q_transition() 
    ...
end

function p_transition() 
    ...
end


# TODO:
# - Support hybrid loss from d3pm
# - Support Unleashing Transformers loss weighting
# - Support MaskGIT sampling
# - Support microsoft paper multinomial+absorbing transitions

