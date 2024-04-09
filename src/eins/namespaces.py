from eins.combination import CustomCombination
from eins.combination import ops as _combination_ops
from eins.common_types import CombinationFunc, ElementwiseFunc, ReductionFunc, TransformationFunc
from eins.elementwise import CustomElementwiseOp
from eins.elementwise import ops as _elementwise_ops
from eins.reduction import CustomReduction
from eins.reduction import Fold as _Fold
from eins.reduction import PowerNorm as _PowerNorm
from eins.reduction import ops as _reduction_ops
from eins.transformation import CustomTransformation
from eins.transformation import PowerNormalize as _PowerNormalize, Softmax as _Softmax
from eins.transformation import Scan as _Scan
from eins.transformation import ops as _transformation_ops


class Combinations:
    """Namespace for built-in combination operations."""

    add = _combination_ops['add']
    hypot = _combination_ops['hypot']
    log_add_exp = _combination_ops['logaddexp']
    maximum = _combination_ops['maximum']
    minimum = _combination_ops['minimum']
    multiply = _combination_ops['multiply']
    bitwise_xor = _combination_ops['bitwise_xor']
    bitwise_and = _combination_ops['bitwise_and']
    bitwise_or = _combination_ops['bitwise_or']

    @staticmethod
    def from_func(func: CombinationFunc) -> CustomCombination:
        """
        Create a user-defined combination operation, from function of signature `func(Array, Array)
        -> Array`.
        """
        return CustomCombination(func=func)

    pass


class ElementwiseOps:
    """Namespace for built-in elementwise operations."""

    abs = _elementwise_ops['abs']
    acos = _elementwise_ops['acos']
    acosh = _elementwise_ops['acosh']
    asin = _elementwise_ops['asin']
    asinh = _elementwise_ops['asinh']
    atan = _elementwise_ops['atan']
    atanh = _elementwise_ops['atanh']
    bitwise_invert = _elementwise_ops['bitwise_invert']
    ceil = _elementwise_ops['ceil']
    conj = _elementwise_ops['conj']
    cos = _elementwise_ops['cos']
    cosh = _elementwise_ops['cosh']
    exp = _elementwise_ops['exp']
    expm1 = _elementwise_ops['expm1']
    floor = _elementwise_ops['floor']
    imag = _elementwise_ops['imag']
    log = _elementwise_ops['log']
    log1p = _elementwise_ops['log1p']
    log2 = _elementwise_ops['log2']
    log10 = _elementwise_ops['log10']
    negative = _elementwise_ops['negative']
    positive = _elementwise_ops['positive']
    real = _elementwise_ops['real']
    round = _elementwise_ops['round']
    sign = _elementwise_ops['sign']
    sin = _elementwise_ops['sin']
    sinh = _elementwise_ops['sinh']
    square = _elementwise_ops['square']
    sqrt = _elementwise_ops['sqrt']
    tan = _elementwise_ops['tan']
    tanh = _elementwise_ops['tanh']
    trunc = _elementwise_ops['trunc']

    @staticmethod
    def from_func(func: ElementwiseFunc) -> CustomElementwiseOp:
        """
        Create a user-defined elementwise operation, from function of signature `func(Array) ->
        Array`.
        """
        return CustomElementwiseOp(func=func)

    pass


class Transformations:
    """Namespace for built-in transformation operations."""

    PowerNormalize = _PowerNormalize
    Softmax = _Softmax
    Scan = _Scan
    sort = _transformation_ops['sort']
    cumulative_sum = _transformation_ops['cumulative_sum']
    normalize = _transformation_ops['normalize']
    l2_normalize = _transformation_ops['l2_normalize']
    l1_normalize = _transformation_ops['l1_normalize']
    inf_normalize = _transformation_ops['inf_normalize']
    quantile = _transformation_ops['quantile']
    min_max_normalize = _transformation_ops['min_max_normalize']

    @staticmethod
    def from_func(func: TransformationFunc) -> CustomTransformation:
        """
        Create a user-defined transformation, from function of signature `func(Array, axis: int) ->
        Array`.
        """
        return CustomTransformation(func=func)


class Reductions:
    """Namespace for built-in reduction operations."""

    PowerNorm = _PowerNorm
    Fold = _Fold
    sum = _reduction_ops['sum']
    mean = _reduction_ops['mean']
    min = _reduction_ops['min']
    max = _reduction_ops['max']
    prod = _reduction_ops['prod']
    norm = _reduction_ops['norm']
    l2_norm = _reduction_ops['l2_norm']
    l1_norm = _reduction_ops['l1_norm']
    inf_norm = _reduction_ops['inf_norm']
    range = _reduction_ops['range']
    ptp = _reduction_ops['ptp']

    @staticmethod
    def from_func(func: ReductionFunc) -> CustomReduction:
        """
        Create a user-defined reduction operation, from function of signature `func(Array, axis:
        int) -> Array` that eliminates the axis specified.
        """
        return CustomReduction(func=func)
