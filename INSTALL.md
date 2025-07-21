## Requirements

If you want to use any Ansys-related features of this project, Ansys Workbench ([free for students](https://www.ansys.com/academic/students/ansys-student)) must be installed on your machine. This requires Microsoft Windows 10 or 11.

## Install from PyPI (recommended)

Once the project has been published on [PyPI](https://pypi.org), you can use it as follows.

### Install Python

Ensure that [Python 3.10](https://www.python.org/downloads/) is installed on your machine. This package has not yet been tested with other Python versions due to compatibility issues with PyAnsys but will be upgraded in the future.

### Create a new Python project

If you want to use OAMC in an existing Python 3.10 project, skip this step.

Create an empty directory for your project, open PowerShell and `cd` into it. Run `py -m venv .venv` on Windows or `python3 -m venv .venv` on Linux and macOS to create a new virtual environment. Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux and macOS. Ensure that your `pip` is up-to-date by running `py -m pip install --upgrade pip` on Windows or `python3 -m pip install --upgrade pip` on Linux and macOS.

### Install OAMC

Install the package with `py -m pip install oamc` on Windows or `python3 -m pip install oamc` on macOS.

### Use OAMC

Have a look at the examples provided in the `examples/` directory.

## Install from Source

### Install uv

This project uses Astral's [uv](https://docs.astral.sh/uv/) for dependency management. If uv is not installed on your machine, you can either follow [these steps](https://docs.astral.sh/uv/getting-started/installation/) or, if you are using Windows, open Command Prompt and run the following command:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

This will launch a new PowerShell instance, override the default execution policy (`Restricted`), and execute the command (`-c`) `irm https://astral.sh/uv/install.ps1 | iex`. PowerShell will then download (`irm` is an alias for `Invoke-RestMethod`) and execute (`iex` is an alias for `Invoke-Expression`) the installation script. No need to worry as we know that [astral.sh](https://astral.sh) can be trusted.

### Install git

If git is not installed on your machine, download the latest installer for your OS from [here](https://git-scm.com/downloads) and install it.

### Clone the repository

Open Git Bash and navigate (`cd`) to the location where you want to clone the repository. Then, run
```
git clone https://github.com/nicolasebeling/oamc.git
```

### Install dependencies

Run
```
uv sync --extra dev
```
to automatically create a virtual environment and install all dependencies (including those for development).

### Activate the virtual environment

Run
```
.venv/Scripts/activate
```
on Windows or
```
.venv/bin/activate
```
on Linux and macOS.

### Run examples

Run
```
uv run examples/001/main.py
```
and so on.

If you run `main.py` from `examples/001/` instead of the root directory of the repository (the Run button of your IDE will likely do this), you have to replace `./examples/001.` with `./` in all relative file paths.

Please make sure to not push modified examples (including paths to local files and directories, for example) or reproducible output files to the repository.

If you have questions, feel free to ask!
