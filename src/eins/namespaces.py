from eins.combination import CustomCombination
from eins.combination import ops as _combination_ops
from eins.common_types import (
    Combination,
    CombinationFunc,
    ElementwiseFunc,
    ElementwiseOp,
    Reduction,
    ReductionFunc,
    Transformation,
    TransformationFunc,
)
from eins.elementwise import Affine as _Affine
from eins.elementwise import CustomElementwiseOp
from eins.elementwise import ops as _elementwise_ops
from eins.reduction import CustomReduction
from eins.reduction import Fold as _Fold
from eins.reduction import PowerNorm as _PowerNorm
from eins.reduction import ops as _reduction_ops
from eins.transformation import CustomTransformation
from eins.transformation import PowerNormalize as _PowerNormalize
from eins.transformation import Scan as _Scan
from eins.transformation import Softmax as _Softmax
from eins.transformation import ops as _transformation_ops


class Combinations:
    """Namespace for built-in combination operations."""

    #: Addition.
    add: Combination = _combination_ops['add']
    #: Hypotenuse: sqrt(x^2 + y^2).
    hypot: Combination = _combination_ops['hypot']
    #: log(exp(x) + exp(y)).
    log_add_exp: Combination = _combination_ops['logaddexp']
    #: Elementwise maximum.
    maximum: Combination = _combination_ops['maximum']
    #: Elementwise minimum.
    minimum: Combination = _combination_ops['minimum']
    #: Multiplication.
    multiply: Combination = _combination_ops['multiply']
    #: Bitwise xor.
    bitwise_xor: Combination = _combination_ops['bitwise_xor']
    #: Bitwise and.
    bitwise_and: Combination = _combination_ops['bitwise_and']
    #: Bitwise or.
    bitwise_or: Combination = _combination_ops['bitwise_or']

    @staticmethod
    def from_func(func: CombinationFunc) -> CustomCombination:
        """
        Create a user-defined combination operation, from function of signature
        `func(Array, Array) → Array`.
        """
        return CustomCombination(func=func)

    pass


class ElementwiseOps:
    """Namespace for built-in elementwise operations."""

    #: Affine transform: Affine(scale, shift) = x * scale + shift.
    Affine: _Affine = _Affine

    #: Absolute value.
    abs: ElementwiseOp = _elementwise_ops['abs']
    #: Inverse cosine.
    acos: ElementwiseOp = _elementwise_ops['acos']
    #: Inverse hyperbolic cosine.
    acosh: ElementwiseOp = _elementwise_ops['acosh']
    #: Inverse sine.
    asin: ElementwiseOp = _elementwise_ops['asin']
    #: Inverse hyperbolic sine.
    asinh: ElementwiseOp = _elementwise_ops['asinh']
    #: Inverse tangent.
    atan: ElementwiseOp = _elementwise_ops['atan']
    #: Inverse hyperbolic tangent.
    atanh: ElementwiseOp = _elementwise_ops['atanh']
    #: Bitwise inversion.
    bitwise_invert: ElementwiseOp = _elementwise_ops['bitwise_invert']
    #: Ceiling.
    ceil: ElementwiseOp = _elementwise_ops['ceil']
    #: Complex conjugate.
    conj: ElementwiseOp = _elementwise_ops['conj']
    #: Cosine.
    cos: ElementwiseOp = _elementwise_ops['cos']
    #: Hyperbolic cosine.
    cosh: ElementwiseOp = _elementwise_ops['cosh']
    #: Exponential.
    exp: ElementwiseOp = _elementwise_ops['exp']
    #: Exponential minus 1.
    expm1: ElementwiseOp = _elementwise_ops['expm1']
    #: Floor.
    floor: ElementwiseOp = _elementwise_ops['floor']
    #: Imaginary part.
    imag: ElementwiseOp = _elementwise_ops['imag']
    #: Logarithm.
    log: ElementwiseOp = _elementwise_ops['log']
    #: Logarithm minus 1.
    log1p: ElementwiseOp = _elementwise_ops['log1p']
    #: Base-2 logarithm.
    log2: ElementwiseOp = _elementwise_ops['log2']
    #: Base-10 logarithm.
    log10: ElementwiseOp = _elementwise_ops['log10']
    #: Negative value.
    negative: ElementwiseOp = _elementwise_ops['negative']
    #: Positive value.
    positive: ElementwiseOp = _elementwise_ops['positive']
    #: Real part.
    real: ElementwiseOp = _elementwise_ops['real']
    #: Round to nearest integer.
    round: ElementwiseOp = _elementwise_ops['round']
    #: Sign of value.
    sign: ElementwiseOp = _elementwise_ops['sign']
    #: Sine.
    sin: ElementwiseOp = _elementwise_ops['sin']
    #: Hyperbolic sine.
    sinh: ElementwiseOp = _elementwise_ops['sinh']
    #: Square.
    square: ElementwiseOp = _elementwise_ops['square']
    #: Square root.
    sqrt: ElementwiseOp = _elementwise_ops['sqrt']
    #: Tangent.
    tan: ElementwiseOp = _elementwise_ops['tan']
    #: Hyperbolic tangent.
    tanh: ElementwiseOp = _elementwise_ops['tanh']
    #: Truncate to integer.
    trunc: ElementwiseOp = _elementwise_ops['trunc']

    @staticmethod
    def from_func(func: ElementwiseFunc) -> CustomElementwiseOp:
        """
        Create a user-defined elementwise operation, from function of signature
        `func(Array) → Array`.
        """
        return CustomElementwiseOp(func=func)

    pass


class Transformations:
    """Namespace for built-in transformation operations."""

    #: Power normalization.
    PowerNormalize: _PowerNormalize = _PowerNormalize
    #: Softmax with optional temperature.
    Softmax: _Softmax = _Softmax
    #: Scanned combination operation: e.g., cumulative sum.
    Scan: _Scan = _Scan
    #: Sort
    sort = _transformation_ops['sort']
    #: Cumulative sum.
    cumulative_sum = _transformation_ops['cumulative_sum']
    #: Normalize to unit standard deviation. Alias of [l2_normalize].
    normalize: Transformation = _transformation_ops['normalize']
    #: Normalize to unit standard deviation.
    l2_normalize: Transformation = _transformation_ops['l2_normalize']
    #: Normalize so the sum of the absolute values is 1.
    l1_normalize: Transformation = _transformation_ops['l1_normalize']
    #: Normalize so the maximum absolute value is 1.
    inf_normalize: Transformation = _transformation_ops['inf_normalize']
    #: Min-max normalization: sets the minimum to 0 and maximum to 1.
    min_max_normalize: Transformation = _transformation_ops['min_max_normalize']

    @staticmethod
    def from_func(func: TransformationFunc) -> CustomTransformation:
        """
        Create a user-defined transformation, from function of signature
        `func(Array, axis: int) → Array`.
        """
        return CustomTransformation(func=func)


class Reductions:
    """Namespace for built-in reduction operations."""

    #: Power norm: 1 is L1 norm, 2 is L2 norm.
    PowerNorm: _PowerNorm = _PowerNorm
    #: Fold operation: e.g., folded add is sum.
    Fold: _Fold = _Fold
    #: Sum.
    sum: Reduction = _reduction_ops['sum']
    #: Mean.
    mean: Reduction = _reduction_ops['mean']
    #: Min.
    min: Reduction = _reduction_ops['min']
    #: Max.
    max: Reduction = _reduction_ops['max']
    #: Product.
    prod: Reduction = _reduction_ops['prod']
    #: Euclidean (L2) norm.
    norm: Reduction = _reduction_ops['norm']
    #: L2 norm.
    l2_norm: Reduction = _reduction_ops['l2_norm']
    #: L1 norm.
    l1_norm: Reduction = _reduction_ops['l1_norm']
    #: Chebyshev norm: maximum absolute value.
    inf_norm: Reduction = _reduction_ops['inf_norm']
    #: Range: max - min.
    range: Reduction = _reduction_ops['range']
    #: Alias of range: max - min.
    ptp: Reduction = _reduction_ops['ptp']

    @staticmethod
    def from_func(func: ReductionFunc) -> CustomReduction:
        """
        Create a user-defined reduction operation, from function of signature `func(Array, axis:
        int) → Array` that eliminates the axis specified.
        """
        return CustomReduction(func=func)
