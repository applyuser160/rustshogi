Installation
============

rustshogi is distributed as a Python package.

Requirements
------------

* Python 3.8 or higher
* Windows, macOS, Linux (x86_64)

Installation with pip
=====================

.. code-block:: bash

   pip install rustshogi

Installing the development version
================================

To install the latest development version:

.. code-block:: bash

   pip install git+https://github.com/applyuser160/rustshogi.git

Building from source
====================

A Rust toolchain is required:

.. code-block:: bash

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Clone the project
   git clone https://github.com/applyuser160/rustshogi.git
   cd rustshogi

   # Build and install
   pip install -e .

Verifying the installation
==========================

To verify that the installation was successful:

.. code-block:: python

   import rustshogi
   print(rustshogi.__version__)
