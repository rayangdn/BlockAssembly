from compas_cra.equilibrium.rbe_pyomo import rbe_solve
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning


def is_stable_rbe(assembly_env, verbose=False):
    # the solver fails if there are no edges, so we are handling this case here
    if assembly_env.graph.number_of_edges() == 0:
        free_nodes = [
            node
            for node in assembly_env.graph.node.values()
            if not node.get("is_support")
        ]
        # no edges and more than 1 free node means the structure is unstable
        return len(free_nodes) == 0, None

    res, res_dict = True, None
    try:
        with SuppressInfeasibleWarning():
            rbe_solve(
                assembly_env,
                mu=assembly_env.mu,
                density=assembly_env.density,
                penalty=False,
                verbose=verbose,
            )
    except (ValueError, IndexError) as e:
        if e.args[0] == "infeasible":
            res, res_dict = False, None
        else:
            print("Warning: Solver failure, assuming unstable")
    return res, res_dict
