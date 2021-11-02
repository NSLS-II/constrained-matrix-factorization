============================
Using Native Constrained NMF
============================

Start by importing Constrained Matrix Factorization.

.. code-block:: python

    import constrainedmf as cmf


For extensibility, each instance of NMF is inherits from a base class.
This class set's up the matrix factorization using the correct shapes and number of components.
It also allows for portions of the matricies to be initialized, and fixed.

Simple NMF
----------

A simple example of NMF can be accomplished with the following, starting from a matrix of 30 different 100 member vectors.
We can constrain the components to be zeros and ones, and allow a third component to approximate the variation.

.. code-block:: python

    import torch

    X = torch.rand(30, 100)
    x_0 = torch.zeros(1, 100)
    x_1 = torch.ones(1, 100)

    model = cmf.nmf.models.NMF(X.shape,
                               3,
                               initial_components=[x_0, x_1],
                               fix_components=[True, True, False])
    H, W = model.fit_transform(X)








.. autoclass:: constrainedmf.nmf.models.NMF
    :members: reconstruct

The underlying class
--------------------

Referring to the base class details some of the constraint functionality available.
The base class could be conceivibly used to construct different reconstruction approaches with NMF; however,
has some required implementations for any child classes.

.. autoclass:: constrainedmf.nmf.models.NMFBase
    :members: get_W_positive, get_H_positive, reconstruct

