import logging
from time import perf_counter as clock

import numpy

from oamc.integrations.ansys.parser import APDLParser
from oamc.integrations.ansys.session import MAPDLSession
from oamc.logging import enable_logging

logger = logging.getLogger(__name__)


class TestSolidModel:
    """Contains test methods for the `SolidModel` class."""

    def test_u(self):
        enable_logging()

        PATH = R".\tests\test_fem\ds.dat"

        start = clock()

        with MAPDLSession(PATH) as session:
            numpy.savetxt(
                "u_ansys.txt",
                session.mapdl.post_processing.nodal_displacement("ALL"),
            )

        logger.info(f"MAPDL took {round(clock() - start, 3)} seconds.")

        start = clock()

        parser = APDLParser(PATH)
        model = parser.get_solid_model()

        numpy.savetxt(
            "u_oamc.txt",
            model.u.reshape(-1, 3),
        )

        logger.info(f"OAMC took {round(clock() - start, 3)} seconds.")
