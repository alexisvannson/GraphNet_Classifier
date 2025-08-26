import torch.nn as nn

def get_loss_function(loss_name="cross_entropy", **kwargs):
    """
    Returns a loss function based on the provided name.
    Args:
        loss_name (str): Name of the loss function. Supported: 
            "cross_entropy", "mse", "bce", "nll", "l1", "smooth_l1"
        **kwargs: Additional keyword arguments for the loss function.
    Returns:
        loss_fn: Instantiated loss function.
    """
    loss_name = loss_name.lower()
    match loss_name:
        case "cross_entropy" | "crossentropy":
            return nn.CrossEntropyLoss(**kwargs)
        case "mse" | "mse_loss" | "mean_squared_error":
            return nn.MSELoss(**kwargs)
        case "bce" | "bce_loss" | "binary_cross_entropy":
            return nn.BCELoss(**kwargs)
        case "nll" | "nll_loss" | "negative_log_likelihood":
            return nn.NLLLoss(**kwargs)
        case "l1" | "l1_loss" | "mae":
            return nn.L1Loss(**kwargs)
        case "smooth_l1" | "smoothl1" | "huber":
            return nn.SmoothL1Loss(**kwargs)
        case _:
            raise ValueError(f"Unsupported loss function: {loss_name}")
