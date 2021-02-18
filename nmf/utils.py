import torch


def normalize(x: torch.Tensor, axis=0) -> torch.Tensor:
    return x / x.sum(axis, keepdim=True)

def scalar_to_vec(x, k, dist='unif', nsig=3):
    '''
    Helper function to generate 1d kernel distributions that integrate
      to a specified scalar. Used in generalizing NMF initialization 
      weights to NMFD.
    ------------
    Parameters
    ------------
    x: float to which the 1d kernel should integrate
    k: integer width of the 1d kernel
    dist: probability distribution ("unif" or "gauss") to use for kernel
    nsig: for Gaussian kernels, number of SDs to span in k steps
    ------------ 
    Returns
    ------------
    numpy array of shape [1,k]
    '''
    if dist == 'unif':
        return np.ones(k) * (x/k)
    
    elif dist == 'gauss':
        t = np.linspace(-nsig, nsig, k+1)
        return np.diff(st.norm.cdf(t)) * x
    
    else:
        raise ValueError('Currently supports only "unif" and "gauss"')