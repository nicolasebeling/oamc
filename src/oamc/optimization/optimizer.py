import logging

from oamc.optimization.optimization_problem import OptimizationProblem

logger = logging.getLogger(__name__)


class Optimizer:
    """

    :param problem: optimization problem to solve
    """

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
