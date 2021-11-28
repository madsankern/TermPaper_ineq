#-----------------------------------------------#
# Author: Mads Anker Nielsen                    #
# Date: 25-11-2021                              #
#                                               #
# Collection of functions used for the project. #
#-----------------------------------------------#

###########
# Imports #
###########
import numpy as np

##############################################
# Density of the Pareto type II distribution #
##############################################
def den_pareto2(x, alpha, mu):

    '''Returns the density of the Pareto type II distribution'''

    den = alpha*(1 + x - mu)**(- (alpha + 1))
    
    return den

##################################
# Acceptance-rejection algorithm #
##################################
def act_rejct(N, c, envelope, target):

    '''Performs rejection sampling of the target distribution'''

    # Define parameter values
    alpha = 1.5
    mu = -0.3

    # Ensure that accepted sample is approx of size N
    sample_size = int(np.round(N*c))

    # Draw uniform numbers to draw from envelop and for accept-rejct step
    rand = np.random.uniform(0, 1, size=(sample_size,2))

    # Draw sample from envelope density
    x = mu + (1 - rand[:,0])**(-1/alpha) - 1

    # Acceptance-Rejection step
    sample = x[rand[:,1]*c*envelope(x) < target(x)]

    return sample

################################
# Compute the Gini coefficient #
################################
def gini(x, w=None):

    '''Compute the gini coefficient for array x'''

    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
