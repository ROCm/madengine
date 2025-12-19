# Build madengine

Clone the madengine repository to your local machine and build it from source by following these steps:

```shell
git clone git@github.com:ROCm/madengine.git
 
# Change folder to madengine
cd madengine
 
# Now run this command from the same directory where pyproject.toml is located:
pip install .
```

## Install from GitHub

You can also directly install the madengine library from the repository.

```shell
pip intall git+https://username:password@github.com/ROCm/madengine.git@main
```

After a successful installation, you can use `pip list`/`pip freeze` to verify that madengine was succesfully installed in your environment.
You can then use the madengine CLI to run containerized models from [MAD](https://github.com/ROCm/MAD). 
