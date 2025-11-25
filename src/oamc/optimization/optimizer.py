import logging

from oamc.optimization.optimization_problem import OptimizationProblem

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Attributes
    ----------
    problem : oamc.optimization.OptimisationProblem
        Optimization problem to solve.
    """

    def __init__(self, problem: OptimizationProblem):
        """
        Parameters
        ----------
        problem : oamc.optimization.OptimisationProblem
            Optimization problem to solve."""
        self.problem = problem
