============
Installation
============

At the command line::

    $ pip install constrained-matrix-factorization



Developer's Instructions
------------------------

Install from github

.. code-block:: bash

    $ python3 -m venv cmf_env
    $ source cmf_env/bin/activate
    $ git clone https://github.com/nsls-ii/constrained-matrix-factorization
    $ cd constrained-matrix-factorization
    $ python -m pip install --upgrade pip wheel
    $ python -m pip install -r requirements-dev.txt
    $ pre-commit install
    $ python -m pip install -e .