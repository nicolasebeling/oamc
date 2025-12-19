"""
Classes
-------
MAPDLSession
"""

import logging
import shutil
from pathlib import Path
from time import perf_counter as timer

import ansys.mapdl.core as mapdl
from ansys.dpf import core as dpf

logger = logging.getLogger(__name__)


class MAPDLSession:
    """Starts and closes an MAPDL session in a context manager.

    Inserts `outres,nload,all` in the solution routine of the provided
    APDL file to ensure that nodal loads are saved to the database.

    Attributes
    ----------
    path : pathlib.Path
        Path to the APDL file.
    mapdl : ansys.mapdl.core.Mapdl
        MAPDL instance.

    Examples
    --------
    ```
    with MAPDLSession("C:/path/to/ds.dat") as session:
        dlist = session.mapdl.dlist()
    ```
    """

    def __init__(self, path: str):
        """Initialize a new instance.

        Parameters
        ----------
        path : str
            Path to the APDL file.
        """

        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.mapdl = None

    def __enter__(self):
        # Insert "outres,nload,all" in the solution routine,
        # so the database contains equivalent nodal loads after solving:
        with open(self.path, "r") as file:
            lines = file.readlines()
        in_outres_block = False
        for index, line in enumerate(lines):
            if line.lower().startswith("outres"):
                if line.lower() == "outres,nload,all\n":
                    break
                in_outres_block = True
            elif in_outres_block:
                lines.insert(index, "outres,nload,all\n")
                break
        with open(self.path, "w") as file:
            file.writelines(lines)

        # Create a directory for working files:
        self.run_location = self.path.parent / "temp"
        self.run_location.mkdir(parents=True, exist_ok=True)

        start = timer()

        # Launch a new MAPDL session:
        self.mapdl = mapdl.launch_mapdl(
            run_location=str(self.run_location),
            loglevel="WARNING",
            override=True,
            cleanup_on_exit=True,
            start_timeout=60,
            port=None,
        )

        logger.info(f"MAPDL launched in {round(timer() - start, 3)} seconds.")

        # Clear the database (not necessary as this is a new instance but safety first):
        self.mapdl.clear()

        # Enter PREP7:
        self.mapdl.prep7()

        start = timer()

        # Read the ds.dat file:
        self.mapdl.input(str(self.path))

        # The model is automatically solved as the ds.dat file contains a SOLVE command.

        logger.info(f"File processed by MAPDL in {round(timer() - start, 3)} seconds.")

        # Exit the current processor:
        self.mapdl.finish()

        # Enter POST1:
        self.mapdl.post1()

        # Read the last data set:
        self.mapdl.set("LAST")

        # Select all entities:
        self.mapdl.allsel("ALL", "ALL")

        # Disable headers in string output such as PRNLD:
        self.mapdl.header("OFF")

        # Maximum page dimensions for continuous output:
        self.mapdl.page(iline=1e9, ichar=132, bline=-1, bchar=240)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        start = timer()
        if self.mapdl is not None:
            self.mapdl.exit()
        dpf.server.shutdown_all_session_servers()
        shutil.rmtree(self.run_location)
        logger.info(
            f"MAPDL exited and temp. files deleted in {round(timer() - start, 3)} seconds."
        )
