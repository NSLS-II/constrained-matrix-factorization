================================
Constrained Matrix Factorization
================================

.. image:: https://img.shields.io/travis/nsls-ii/constrained-matrix-factorization.svg
        :target: https://travis-ci.org/nsls-ii/constrained-matrix-factorization

.. image:: https://img.shields.io/pypi/v/constrained-matrix-factorization.svg
        :target: https://pypi.python.org/pypi/constrained-matrix-factorization


Advancements on non-negative matrix factorization in PyTorch, with crystallography as a primary use case. 

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://nsls-ii.github.io/constrained-matrix-factorization.

Features
--------

* TODO

Developer's Instructions
------------------------

Install from github::

    $ python3 -m venv nmf_env
    $ source nmf_env/bin/activate
    $ git clone https://github.com/maffettone/constrained-matrix-factorization
    $ cd constrained-matrix-factorization
    $ python -m pip install --upgrade pip wheel
    $ python -m pip install -r requirements-dev.txt
    $ pre-commit install
    $ python -m pip install -e .

