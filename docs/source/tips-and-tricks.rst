===============
Tips and Tricks
===============


Choosing a number of components
-------------------------------
**SPOILER ALERT**: There may not be one right answer.
Increasing the number of components will in general increase the strength of the reconstruction.
Take a look at the "elbow method" for knowing where increasing the number of components starts to produce diminishing
returns: :code:`constrainedmf.nmf.utils.sweep_components`.

Choosing constraints
--------------------
Constraints can be specified *a priori* based on system knowledge, such as known material states, or in a data-driven manner during the experiment.
For diffraction, one *a priori* approach is to use simulated patterns of known or anticipated phases as constraints, forcing the unconstrained components to describe new or unknown phases.
Alternatively, experimental patterns can be drawn directly from the dataset can be used as fundamental components.
This is a viable approach for a variable temperature series where the end members are expected to be distinct.
This approach is implemented in :code:`constrainedmf.nmf.utils.iterative_nmf`.

Increasing speed!
-----------------
The core objective of this package is speed for on-the-fly analysis.
If/when you start to encounter circumstances where your datasets are growing so large that the implementation is
too slow for your use case, consider the following:

    1. Use CUDA acceleration by setting :code:`device="cuda"`, if you have an available GPU.
    2. Don't be afraid to downsample datasets. Fit the NMF to those fewer data points, then transform the full dataset using the learned components.
    3. Select a region on interest! If your spectrum contains 10,000 floats and only 1000 are non-zero, use this ROI to feed the model and lessen the load.


