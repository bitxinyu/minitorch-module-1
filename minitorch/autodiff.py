from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")
    a = f(*(vals[:arg]+ (vals[arg] + epsilon, ) + vals[arg+1:]))
    b = f(*(vals[:arg]+ (vals[arg] - epsilon, ) + vals[arg+1:]))
    return  (a-b) / ( 2 * epsilon)
    


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    topo_order = []
    seen = set()
    def DFS(variable: Variable):
        if variable.unique_id in seen:
            return
        if variable.is_constant():
            return 
        if not variable.is_leaf():
            parents = variable.parents
            for p in parents:
                DFS(p)
        seen.add(variable.unique_id)
        topo_order.insert(0, variable)
    DFS(variable)
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    topo_order_variables = topological_sort(variable)
    var_id_2_deriv = {variable.unique_id: deriv}
    for var in topo_order_variables:
        if var.is_leaf():
            var.accumulate_derivative(var_id_2_deriv[var.unique_id])
        else:
            back = var.chain_rule(var_id_2_deriv[var.unique_id])
            # back [(varable, deriv)...]
            for b_var, b_deriv in back:
                if b_var.unique_id in var_id_2_deriv:
                    var_id_2_deriv[b_var.unique_id] += b_deriv
                else:
                    var_id_2_deriv[b_var.unique_id] = b_deriv



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
