.. Packaging Scientific Python documentation master file, created by
   sphinx-quickstart on Thu Jun 28 12:35:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Constrained Matrix Factorization Documentation
==============================================
.. warning::

   This is currently under rapid development, the API may change at
   any time. We suggest paying careful attention to the version number you are using.

Constrained Matrix Factorization (CMF) comes as an advancement on Non-negative Matrix Factorization (NMF).
Initially called constrained non-negative matrix factorization, it was recognized that this was
redundant, as the non-negativity is already a constraint.
The goal for this package is to produce rapid matrix factorization approaches with effective constraints
as related to beamline science. We use PyTorch as a backend to enable GPU acceleration and provide constraints
*via* gradient management.

Total scattering (x-ray diffraction and pair distribution function) analysis,
was used as a primary example with rigid constraints in `Applied Physics Reviews <https://doi.org/10.1063/5.0052859>`_
(`arXiv version <https://arxiv.org/abs/2104.00864>`_).


**Please feel free to file bug reports and feature requests *via* GitHub Issues!**

.. toctree::
   :maxdepth: 2

   installation
   nativeCMF
   companions
   examples
   tips-and-tricks
   release-history
   min_versions