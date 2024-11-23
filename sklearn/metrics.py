
import numpy as np
from scipy.special import xlogy
from .common import column_or_1d, _weighted_percentile, _check_reg_targets, _find_matching_floating_dtype

xp = np
_average = np.average



def _assemble_r2_explained_variance(
    numerator, denominator, n_outputs, multioutput, force_finite, xp=xp, device=None
):
    """Common part used by explained variance score and :math:`R^2` score."""
    dtype = numerator.dtype

    nonzero_denominator = denominator != 0

    if not force_finite:
        # Standard formula, that may lead to NaN or -Inf
        output_scores = 1 - (numerator / denominator)
    else:
        nonzero_numerator = numerator != 0
        # Default = Zero Numerator = perfect predictions. Set to 1.0
        # (note: even if denominator is zero, thus avoiding NaN scores)
        output_scores = xp.ones([n_outputs], device=device, dtype=dtype)
        # Non-zero Numerator and Non-zero Denominator: use the formula
        valid_score = nonzero_denominator & nonzero_numerator

        output_scores[valid_score] = 1 - (
            numerator[valid_score] / denominator[valid_score]
        )

        # Non-zero Numerator and Zero Denominator:
        # arbitrary set to 0.0 to avoid -inf scores
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # Passing None as weights to np.average results is uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            if not xp.any(nonzero_denominator):
                # All weights are zero, np.average would raise a ZeroDiv error.
                # This only happens when all y are constant (or 1-element long)
                # Since weights are all equal, fall back to uniform weights.
                avg_weights = None
    else:
        avg_weights = multioutput

    result = _average(output_scores, weights=avg_weights)
    if result.size == 1:
        return float(result)
    return result


def explained_variance_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)

    y_diff_avg = np.average(y_true - y_pred, weights=sample_weight, axis=0)
    numerator = np.average(
        (y_true - y_pred - y_diff_avg) ** 2, weights=sample_weight, axis=0
    )

    y_true_avg = np.average(y_true, weights=sample_weight, axis=0)
    denominator = np.average((y_true - y_true_avg) ** 2, weights=sample_weight, axis=0)

    return _assemble_r2_explained_variance(
        numerator=numerator,
        denominator=denominator,
        n_outputs=y_true.shape[1],
        multioutput=multioutput,
        force_finite=force_finite,
        xp=xp,
        # TODO: update once Array API support is added to explained_variance_score.
        device=None,
    )


def max_error(y_true, y_pred):
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None, xp=xp)
    if y_type == "continuous-multioutput":
        raise ValueError("Multioutput not supported in max_error")
    return xp.max(xp.abs(y_true - y_pred))

def mean_absolute_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
    xp=np
):
    input_arrays = [y_true, y_pred, sample_weight, multioutput]
    dtype = _find_matching_floating_dtype(y_true, y_pred, sample_weight, xp=xp)

    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )

    output_errors = _average(
        xp.abs(y_pred - y_true), weights=sample_weight, axis=0, xp=xp
    )
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    mean_absolute_error = _average(output_errors, weights=multioutput)

    return float(mean_absolute_error)


def mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
):
    dtype = _find_matching_floating_dtype(y_true, y_pred, xp=xp)

    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )
    output_errors = _average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to _average: uniform mean
            multioutput = None

    mean_squared_error = _average(output_errors, weights=multioutput)

    return float(mean_squared_error)

def mean_squared_log_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
):
    dtype = _find_matching_floating_dtype(y_true, y_pred, xp=xp)

    _, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )

    if xp.any(y_true <= -1) or xp.any(y_pred <= -1):
        raise ValueError(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain values less than or equal to -1."
        )

    return mean_squared_error(
        xp.log1p(y_true),
        xp.log1p(y_pred),
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


def median_absolute_error(
    y_true, y_pred, *, multioutput="uniform_average", sample_weight=None
):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    if sample_weight is None:
        output_errors = np.median(np.abs(y_pred - y_true), axis=0)
    else:
        output_errors = _weighted_percentile(
            np.abs(y_pred - y_true), sample_weight=sample_weight
        )
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
    device=None
):

    dtype = _find_matching_floating_dtype(y_true, y_pred, sample_weight, xp=xp)

    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight, dtype=dtype)
        weight = sample_weight[:, None]
    else:
        weight = 1.0

    numerator = xp.sum(weight * (y_true - y_pred) ** 2, axis=0)
    denominator = xp.sum(
        weight * (y_true - _average(y_true, axis=0, weights=sample_weight, xp=xp)) ** 2,
        axis=0,
    )

    return _assemble_r2_explained_variance(
        numerator=numerator,
        denominator=denominator,
        n_outputs=y_true.shape[1],
        multioutput=multioutput,
        force_finite=force_finite,
        xp=xp,
        device=device,
    )

def _mean_tweedie_deviance(y_true, y_pred, sample_weight, power):
    """Mean Tweedie deviance regression loss."""
    p = power
    zero = xp.asarray(0, dtype=y_true.dtype)
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        dev = 2 * (
            xp.pow(xp.where(y_true > 0, y_true, zero), xp.asarray(2 - p))
            / ((1 - p) * (2 - p))
            - y_true * xp.pow(y_pred, xp.asarray(1 - p)) / (1 - p)
            + xp.pow(y_pred, xp.asarray(2 - p)) / (2 - p)
        )
    elif p == 0:
        # Normal distribution, y and y_pred any real number
        dev = (y_true - y_pred) ** 2
    elif p == 1:
        # Poisson distribution
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)
    elif p == 2:
        # Gamma distribution
        dev = 2 * (xp.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        dev = 2 * (
            xp.pow(y_true, xp.asarray(2 - p)) / ((1 - p) * (2 - p))
            - y_true * xp.pow(y_pred, xp.asarray(1 - p)) / (1 - p)
            + xp.pow(y_pred, xp.asarray(2 - p)) / (2 - p)
        )
    return float(_average(dev, weights=sample_weight))

def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[xp.float64, xp.float32], xp=xp
    )
    if y_type == "continuous-multioutput":
        raise ValueError("Multioutput not supported in mean_tweedie_deviance")

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = sample_weight[:, np.newaxis]

    message = f"Mean Tweedie deviance error with power={power} can only be used on "
    if power < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        if xp.any(y_pred <= 0):
            raise ValueError(message + "strictly positive y_pred.")
    elif power == 0:
        # Normal, y and y_pred can be any real number
        pass
    elif 1 <= power < 2:
        # Poisson and compound Poisson distribution, y >= 0, y_pred > 0
        if xp.any(y_true < 0) or xp.any(y_pred <= 0):
            raise ValueError(message + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        # Gamma and Extreme stable distribution, y and y_pred > 0
        if xp.any(y_true <= 0) or xp.any(y_pred <= 0):
            raise ValueError(message + "strictly positive y and y_pred.")
    else:  # pragma: nocover
        # Unreachable statement
        raise ValueError

    return _mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=power
    )
