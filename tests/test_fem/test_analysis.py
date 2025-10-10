import logging
from time import perf_counter as timer

import numpy

from oamc.fem.readers import DSReader
from oamc.logging import enable_logging

logger = logging.getLogger(__name__)


class TestAnalysis:
    """
    Contains test methods for the Analysis class.
    """

    def test_u(self):
        enable_logging()

        with DSReader(R".\tests\test_fea\ds.dat") as reader:
            analysis = reader.get_analysis()

            start = timer()
            numpy.savetxt(
                "displacements.txt",
                analysis.u,
            )
            print(timer() - start)

            start = timer()
            numpy.savetxt(
                "nodal_displacement.txt",
                reader.mapdl.post_processing.nodal_displacement("ALL"),
            )
            print(timer() - start)
