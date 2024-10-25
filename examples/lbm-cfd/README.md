# LBM-CFD

Lattice-Boltzmann Method 2D Computational Fluid Dynamics Simulation

Code is parallelized using MPI


### MPI
Install desired version of MPI (e.g. OpenMPI, MPICH, etc.)


### Python Virtual Environment
```
cd lbm-cfd
python -m venv ./.venv
source .venv/bin/activate
pip install setuptools
pip install numpy
pip install opencv-python
pip install trame
pip install trame-vuetify
pip install trame-rca
pip install --no-binary :all: --compile mpi4py
```


### Ascent Install and Build (with MPI)
```
git clone --recursive https://github.com/alpine-dav/ascent.git
cd ascent
```

Edit "./scripts/build_ascent/build_ascent.sh" to change mfem version from 4.6 to 4.7

Make sure that the Python virtual environment created in the previous step is activated

```
env enable_python=ON enable_mpi=ON prefix=<ascent_install_dir> ./scripts/build_ascent/build_ascent.sh
```


### Build LBM-CFD application
With Ascent support:
```
env ASCENT_DIR=<ascent_install_dir>/install/ascent-checkout make
```

Without Ascent support:
```
make
```


### Run Trame server
Make sure that the Python virtual environment created in the previous step is activated

```
python trame_app.py --host 0.0.0.0 --port <port> --server --timeout 0
```


### Run LBM-CFD application
Make sure that the Python virtual environment created in the previous step is activated

Ensure Trame server is up and running, and you have opened the web page to the Trame application

Then run the following (note: you may need to specify different versions for Python and Conduit - check your install)
```
PYTHON_SITE_PKG="<python_virtual_env_path>/lib/python3.12/site-packages"
ASCENT_DIR="<ascent_install_dir>/install"
export PYTHONPATH=$PYTHONPATH:PYTHON_SITE_PKG:$ASCENT_DIR/ascent-checkout/python-modules/:$ASCENT_DIR/conduit-v0.9.2/python-modules/

mpiexec -np <num_procs> ./bin/lbmcfd
```
