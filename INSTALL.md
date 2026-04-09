# Install

## Requirements

For testing the `oamc.fem` subpackage against results from Ansys, Ansys Workbench, which is [free for students](https://www.ansys.com/academic/students/ansys-student), must be installed on your machine. This requires Microsoft Windows 10 or 11. Ansys Workbench is also required if you want to use OAMC with your own models, since [Ansys Parametric Design Language (APDL)](https://www.ansys.com/blog/what-is-apdl) is currently the only supported mesh/model format.

## Install from PyPI (recommended)

### Install Python

Ensure that [Python 3.13](https://www.python.org/downloads/) or later is installed on your machine.

### Create a Python project

If you want to use OAMC in an existing project, `cd` into its root directory and go to the [next step](#install-oamc).

Else, create an empty directory for your project, open your terminal and `cd` into it. Run

```powershell
py -m venv .venv
```

on Windows or

```bash
python3 -m venv .venv
```

on Linux and macOS to create a new virtual environment. Activate it with

```powershell
.venv\Scripts\activate
```

on Windows or

```bash
source .venv/bin/activate
```

on Linux and macOS. Ensure that `pip` is up-to-date by running

```powershell
py -m pip install --upgrade pip
```

on Windows or

```bash
python3 -m pip install --upgrade pip`
```

on Linux and macOS. You can also use any other package manager of course.

### Install OAMC

Install the package with

```powershell
py -m pip install oamc
```

on Windows or

```bash
python3 -m pip install oamc
```

on Linux and macOS (again, you can also use any other package manager). If you want to run tests, install `oamc[test]` instead of just `oamc`.

Check your installation by running

```powershell
py -m "from oamc.constants import BANNER; print(BANNER);"
```

on Windows or

```bash
python3 -m "from oamc.constants import BANNER; print(BANNER);"
```

on Linux and macOS.

### Get started

Read the [guide](https://oamc.readthedocs.io/stable/guide.html) or have a look at the `examples/` directory on [GitHub](https://github.com/nicolasebeling/oamc) to see how OAMC can be used in practice.

## Install from source

### Install uv

This project uses Astral's [uv](https://docs.astral.sh/uv/) for dependency management. If uv is not installed on your machine, you can either follow [these steps](https://docs.astral.sh/uv/getting-started/installation/) or, if you are using Windows, open Command Prompt and run the following command:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

This will launch a new PowerShell instance, override the default execution policy (`Restricted`), and execute the command (`-c`) `irm https://astral.sh/uv/install.ps1 | iex`. PowerShell will then download (`irm` is an alias for `Invoke-RestMethod`) and execute (`iex` is an alias for `Invoke-Expression`) the installation script. No need to worry as we know that [astral.sh](https://astral.sh) can be trusted.

### Install git

If git is not installed on your machine, [download](https://git-scm.com/downloads) the latest installer for your OS and install git.

### Clone the repository

Open Git Bash and navigate (`cd`) to the location where you want to clone the repository. Then, run

```bash
git clone https://github.com/nicolasebeling/oamc.git
```

### Install dependencies

Run

```bash
uv venv
```

and

```bash
uv sync
```

to automatically create a virtual environment and install essential dependencies. Add

- `--extra test` to install tools for testing such as `pytest` and `pyansys` (using `pyansys` for testing requires a licensed Ansys Mechanical installation),
- `--extra doc` to install documentation tools such as `sphinx`,
- `--group dev` to install development tools such as `ruff`,
- `--all-extras` and `--al-groups` to install all listed extras and groups, respectively.

### Activate the virtual environment

Run

```powershell
.venv\Scripts\activate
```

on Windows or

```bash
source ./venv/bin/activate
```

on Linux and macOS.

### Get started

Run

```bash
uv run examples/1/main.py
```

and so on. Examples are numbered consecutively. `examples/README.md` contains a short description of each.

If you contribute, make sure to not push modified examples (including paths to local files and directories, for example) or reproducible output files to the repository.

If you have questions, feel free to ask!
