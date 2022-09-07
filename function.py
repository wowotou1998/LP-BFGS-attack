from typing import List, Optional
from torch import Tensor
from collections import namedtuple
import torch
import torch.autograd as autograd
from torch._vmap_internals import _vmap

__all__ = ['ScalarFunction', ]

from functools import reduce
import torch
from torch.optim import Optimizer


class LinearOperator:
    """A generic linear operator to use with Minimizer"""
    def __init__(self, matvec, shape, dtype=torch.float, device=None):
        self.rmv = matvec
        self.mv = matvec
        self.shape = shape
        self.dtype = dtype
        self.device = device


class Minimizer(Optimizer):
    """A general-purpose PyTorch optimizer for unconstrained function
    minimization.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Parameters
    ----------
    params : iterable
        An iterable of :class:`torch.Tensor` s. Specifies what Tensors
        should be optimized.
    method : str
        Minimization method (algorithm) to use. Must be one of the methods
        offered in :func:`torchmin.minimize()`. Defaults to 'bfgs'.
    **minimize_kwargs : dict
        Additional keyword arguments that will be passed to
        :func:`torchmin.minimize()`.

    """
    def __init__(self,
                 params,
                 method='bfgs',
                 **minimize_kwargs):
        assert isinstance(method, str)
        method_ = method.lower()

        self._hessp = self._hess = False
        if method_ in ['bfgs', 'l-bfgs', 'cg']:
            pass
        elif method_ in ['newton-cg', 'trust-ncg', 'trust-krylov']:
            self._hessp = True
        elif method_ in ['newton-exact', 'dogleg', 'trust-exact']:
            self._hess = True
        else:
            raise ValueError('Unknown method {}'.format(method))

        defaults = dict(method=method_, **minimize_kwargs)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Minimizer doesn't support per-parameter options")

        self._nfev = [0]
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._closure = None
        self._result = None

    @property
    def nfev(self):
        return self._nfev[0]

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_param(self):
        params = []
        for p in self._params:
            if p.data.is_sparse:
                p = p.data.to_dense().view(-1)
            else:
                p = p.data.view(-1)
            params.append(p)
        return torch.cat(params)

    def _gather_flat_grad(self):
        grads = []
        for p in self._params:
            if p.grad is None:
                g = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                g = p.grad.to_dense().view(-1)
            else:
                g = p.grad.view(-1)
            grads.append(g)
        return torch.cat(grads)

    def _set_flat_param(self, value):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.copy_(value[offset:offset+numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    def closure(self, x):
        from function import sf_value

        assert self._closure is not None
        self._set_flat_param(x)
        with torch.enable_grad():
            f = self._closure()
            f.backward(create_graph=self._hessp or self._hess)
            grad = self._gather_flat_grad()

        grad_out = grad.detach().clone()
        hessp = None
        hess = None
        if self._hessp or self._hess:
            grad_accum = grad.detach().clone()
            def hvp(v):
                assert v.shape == grad.shape
                grad.backward(gradient=v, retain_graph=True)
                output = self._gather_flat_grad().detach() - grad_accum
                grad_accum.add_(output)
                return output

            numel = self._numel()
            if self._hessp:
                hessp = LinearOperator(hvp, shape=(numel, numel),
                                       dtype=grad.dtype, device=grad.device)
            if self._hess:
                eye = torch.eye(numel, dtype=grad.dtype, device=grad.device)
                hess = torch.zeros(numel, numel, dtype=grad.dtype, device=grad.device)
                for i in range(numel):
                    hess[i] = hvp(eye[i])

        return sf_value(f=f.detach(), grad=grad_out.detach(), hessp=hessp, hess=hess)

    def dir_evaluate(self, x, t, d):
        from function import de_value

        self._set_flat_param(x + d.mul(t))
        with torch.enable_grad():
            f = self._closure()
        f.backward()
        grad = self._gather_flat_grad()
        self._set_flat_param(x)

        return de_value(f=float(f), grad=grad)

    @torch.no_grad()
    def step(self, closure):
        """Perform an optimization step.

        The function "closure" should have a slightly different
        form vs. the PyTorch standard: namely, it should not include any
        `backward()` calls. Backward steps will be performed internally
        by the optimizer.

        >>> def closure():
        >>>    optimizer.zero_grad()
        >>>    output = model(input)
        >>>    loss = loss_fn(output, target)
        >>>    # loss.backward() <-- skip this step!
        >>>    return loss

        Parameters
        ----------
        closure : callable
            A function that re-evaluates the model and returns the loss.

        """
        from minimize import minimize

        # sanity check
        assert len(self.param_groups) == 1

        # overwrite closure
        closure_ = closure
        def closure():
            self._nfev[0] += 1
            return closure_()
        self._closure = closure

        # get initial value
        x0 = self._gather_flat_param()

        # perform parameter update
        kwargs = {k:v for k,v in self.param_groups[0].items() if k != 'params'}
        self._result = minimize(self, x0, **kwargs)

        # set final value
        self._set_flat_param(self._result.x)

        return self._result.fun

# scalar function result (value)
# 定义一个namedtuple类型 sf_value，并包含'f', 'grad', 'hessp', 'hess'属性. 这个tuple包含了一系列有关于函数f在x处的各种数值
# sf_value = namedtuple('sf_value', ['f', 'grad', 'hessp', 'hess'])
sf_value = namedtuple('sf_value', ['f', 'grad', 'hessp', 'hess'])

# directional evaluate result
de_value = namedtuple('de_value', ['f', 'grad'])

# vector function result (value)
vf_value = namedtuple('vf_value', ['f', 'jacp', 'jac'])


@torch.jit.script
class JacobianLinearOperator(object):
    def __init__(self,
                 x: Tensor,
                 f: Tensor,
                 gf: Optional[Tensor] = None,
                 gx: Optional[Tensor] = None,
                 symmetric: bool = False) -> None:
        self.x = x
        self.f = f
        self.gf = gf
        self.gx = gx
        self.symmetric = symmetric
        # tensor-like properties
        self.shape = (x.numel(), x.numel())
        self.dtype = x.dtype
        self.device = x.device

    def mv(self, v: Tensor) -> Tensor:
        if self.symmetric:
            return self.rmv(v)
        assert v.shape == self.x.shape
        gx, gf = self.gx, self.gf
        assert (gx is not None) and (gf is not None)
        outputs: List[Tensor] = [gx]
        inputs: List[Tensor] = [gf]
        grad_outputs: List[Optional[Tensor]] = [v]
        jvp = autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)[0]
        if jvp is None:
            raise Exception
        return jvp

    def rmv(self, v: Tensor) -> Tensor:
        assert v.shape == self.f.shape
        outputs: List[Tensor] = [self.f]
        inputs: List[Tensor] = [self.x]
        grad_outputs: List[Optional[Tensor]] = [v]
        vjp = autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)[0]
        if vjp is None:
            raise Exception
        return vjp


class ScalarFunction(object):
    """Scalar-valued objective function with autograd backend.

    This class provides a general-purpose objective wrapper which will
    compute first- and second-order derivatives via autograd as specified
    by the parameters of __init__.
    """

    def __new__(cls, fun, x_shape, hessp=False, hess=False, twice_diffable=True):
        if isinstance(fun, Minimizer):
            assert fun._hessp == hessp
            assert fun._hess == hess
            return fun
        return super(ScalarFunction, cls).__new__(cls)

    def __init__(self, fun, x_shape, hessp=False, hess=False, twice_diffable=True):
        self._fun = fun
        self._x_shape = x_shape
        self._hessp = hessp
        self._hess = hess
        self._I = None
        self._twice_diffable = twice_diffable
        self.nfev = 0

    def fun(self, x):
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        f = self._fun(x)
        if f.numel() != 1:
            raise RuntimeError('ScalarFunction was supplied a function '
                               'that does not return scalar outputs.')
        self.nfev += 1

        return f

    def closure(self, x):
        """Evaluate the function, gradient, and hessian/hessian-product

        This method represents the core function call. It is used for
        computing newton/quasi newton directions, etc.
        """
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
            grad = autograd.grad(f, x, create_graph=self._hessp or self._hess)[0]
        if (self._hessp or self._hess) and grad.grad_fn is None:
            raise RuntimeError('A 2nd-order derivative was requested but '
                               'the objective is not twice-differentiable.')
        hessp = None
        hess = None
        if self._hessp:
            hessp = JacobianLinearOperator(x, grad, symmetric=self._twice_diffable)
        if self._hess:
            if self._I is None:
                self._I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            hvp = lambda v: autograd.grad(grad, x, v, retain_graph=True)[0]
            hess = _vmap(hvp)(self._I)

        return sf_value(f=f.detach(), grad=grad.detach(), hessp=hessp, hess=hess)

    def dir_evaluate(self, x, t, d):
        """Evaluate a direction and step size.

        We define a separate "directional evaluate" function to be used
        for strong-wolfe line search. Only the function value and gradient
        are needed for this use case, so we avoid computational overhead.
        """
        x = x + d.mul(t)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
        grad = autograd.grad(f, x)[0]

        return de_value(f=float(f), grad=grad)
