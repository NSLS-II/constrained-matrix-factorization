=============================
Wrappers and Companion Agents
=============================

While the functionality of :code:`nmf.models` is all that is necessary to perform
constrained matrix factorization, it is worth developing some task specific agents
that create reports or characterize data based on the CMF results.

We divide this into two areas:

    1. The wrappers of CMF that do some regular preprocessing or specific deployment that might be
       pertinent to a single class of experiments.
    2. How the agent that depends on this wrapper interfaces with your experimental workflow
       `(see bluesky-adaptive) <https://blueskyproject.io/bluesky-adaptive/>`_.

Experiment specific wrappers
----------------------------

Total scattering: diffraction and pair distribution function
*************************************************************
We use the I(q) function nomenclature as a placeholder for f(x), but this could easily be swapped for datasets
like F(q), I(2theta), or G(r).

- .. autofunction:: constrainedmf.wrappers.scattering.decomposition

A handy alternative that makes use of :code:`constrainedmf.nmf.utils.iterative_nmf` is the iterative decomposition
that makes use of the members the dataset to chose constraints for the NMF. This process occurs iteratively, and
uses the dataset end-members, then the members that produce the maximal information gain of the model.
Commonly, this function will be used with multiple numbers of components in parallel, feeding the scientist a
visualization of their dataset in real-time.

- .. autofunction:: constrainedmf.wrappers.scattering.iterative_decomposition


Companion agents for bluesky
----------------------------

Building useful companion agents is an active area of development.
For more insights in how to deploy cmf, check out this repository for
`experiments at 28-ID PDF <https://github.com/NSLS-II-PDF/federation-of-agents/blob/main/federation/nmf/base.py>`_.
