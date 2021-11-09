================================
Constrained Matrix Factorization
================================

.. image:: https://img.shields.io/github/workflow/status/nsls-ii/constrained-matrix-factorization/Unit%20Tests
        :target: https://github.com/nsls-ii/constrained-matrix-factorization/actions?query=workflow%3A%22Unit+Tests%22+branch%3Amain

.. image:: https://img.shields.io/pypi/v/constrained-matrix-factorization.svg
        :target: https://pypi.python.org/pypi/constrained-matrix-factorization


Advancements on non-negative matrix factorization in PyTorch, with crystallography as a primary use case. 

* Free software: 3-clause BSD license
* Documentation: https://nsls-ii.github.io/constrained-matrix-factorization.

Features
--------

* Torch enabled non-negative matrix factorization
* Optional input components and/or weights
* Rigidly constrained specific components or weights
* Plotting utilities
* Utilities for "elbow method" and automatic iterative cmf



Developer's Instructions
------------------------

Install from github::

    $ python3 -m venv nmf_env
    $ source nmf_env/bin/activate
    $ git clone https://github.com/nsls-ii/constrained-matrix-factorization
    $ cd constrained-matrix-factorization
    $ python -m pip install --upgrade pip wheel
    $ python -m pip install -r requirements-dev.txt
    $ pre-commit install
    $ python -m pip install -e .



Original Publication
--------------------
This work was originally published on `the arXiv here of this paper <https://arxiv.org/abs/2104.00864>`_.
It has since been peer reviewd and published in `Applied Physics Reviews <https://doi.org/10.1063/5.0052859>`_.

Abstract
========
Non-negative Matrix Factorization (NMF) methods offer an appealing unsupervised learning method for real-time analysis of streaming spectral data in time-sensitive data collection, such as in situ characterization of materials. However, canonical NMF methods are optimized to reconstruct a full dataset as closely as possible, with no underlying requirement that the reconstruction produces components or weights representative of the true physical processes. In this work, we demonstrate how constraining NMF weights or components, provided as known or assumed priors, can provide significant improvement in revealing true underlying phenomena. We present a PyTorch based method for efficiently applying constrained NMF and demonstrate this on several synthetic examples. When applied to streaming experimentally measured spectral data, an expert researcher-in-the-loop can provide and dynamically adjust the constraints. This set of interactive priors to the NMF model can, for example, contain known or identified independent components, as well as functional expectations about the mixing of components. We demonstrate this application on measured X-ray diffraction and pair distribution function data from in situ beamline experiments. Details of the method are described, and general guidance provided to employ constrained NMF in extraction of critical information and insights during in situ and high-throughput experiments.
