===============
Release History
===============

NEXT
----

- Address scope and convenience imports. The nmf module is accesible at the top level, as well as the NMF class.
  :code:`cmf.nmf...` and :code:`cmf.NMF(...)`.
- More flexibility at torch level with constraints. Allows for partial initialization of componenents and weights, and
  partial specification of constraints.

v0.1.1 (2021-11-?)
-------------------

- Refactor of companions and wrappers: wrappers wrap functionality; companions have :code:`ask, tell, report` methods.
- CUDA and GPU Functionality added to base classes, utility functions, and total scattering
- Required kwargs on functions, instead of allowing positional.
- Numerical instability patch for when positive comps (mu denominator in multiplicative update) are 0.
- Added much more documentation.

v0.1.0 (2021-11-01)
-------------------
This initial release marks the transfer of this repository to the NSLS-II organization.
Initial publication details can be found on  `the arXiv <https://arxiv.org/abs/2104.00864>`_,
with the peer reviewed version accepted for publication in Applied Physics Reviews.