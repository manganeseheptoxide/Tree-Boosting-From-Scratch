import numpy as np

class LossFunction:
    def __init__(self):
        self.loss = self.loss_calculation
        self.gradient = self.gradient_calculation
        self.hessian = self.hessian_calculation

    def loss_calculation(self, actual, predicted):
        raise NotImplementedError("Loss function not implemented!")

    def gradient_calculation(self, actual, predicted): # Derivative of loss function with respect to predicted values (Technically negative gradient)
        raise NotImplementedError("Gradient calculation not implemented!")
    
    def hessian_calculation(self, actual, predicted):
        raise NotImplementedError("Hessian calculation not implemented!")
    

class SSR(LossFunction):
    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        return np.sum((y_actual - y_predicted) ** 2)

    def gradient_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        # Correct derivative with respect to y_predicted:
        return -2 * (y_actual - y_predicted)
    
    def hessian_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        # Second derivative is constant (2)
        return np.full_like(y_actual, 2.0)


class QuantileLoss(LossFunction):
    def __init__(self, tau=0.9):
        super().__init__()
        self.tau = tau  # Quantile level (0 < tau < 1)
    
    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.sum(np.maximum(self.tau * residual, (self.tau - 1) * residual))

    def gradient_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        # For quantile loss: if y_actual > y_predicted, derivative with respect to y_predicted is -tau,
        # else it's -(tau-1)
        return -np.where(y_actual > y_predicted, self.tau, self.tau - 1)
    
    def hessian_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        # Piecewise linear loss => second derivative is 0 almost everywhere.
        return np.zeros_like(y_actual)


class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta  # Threshold for switching between MSE and MAE

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        mask = np.abs(residual) <= self.delta
        return np.sum(np.where(mask, 0.5 * residual**2, self.delta * (np.abs(residual) - 0.5 * self.delta)))

    def gradient_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        # In the quadratic region: derivative = -residual; in the linear region: derivative = -delta*sign(residual)
        return -np.where(np.abs(residual) <= self.delta, residual, self.delta * np.sign(residual))
    
    def hessian_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        # Second derivative is 1 in quadratic region, 0 otherwise.
        return np.where(np.abs(residual) <= self.delta, 1.0, 0.0)


class AsymmetricHuberLoss(LossFunction):
    def __init__(self, delta=1.0, alpha=3.0):  # alpha > 1 penalizes underestimation more
        super().__init__()
        self.delta = delta
        self.alpha = alpha  

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        mask = np.abs(residual) <= self.delta
        return np.sum(
            np.where(mask, 0.5 * residual**2, 
                     np.where(residual > 0, self.alpha * self.delta * (np.abs(residual) - 0.5 * self.delta),
                              self.delta * (np.abs(residual) - 0.5 * self.delta)))
        )

    def gradient_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        # In the quadratic region: derivative = -residual.
        # In the linear region, if residual > 0: derivative = -alpha * delta,
        # else: derivative = delta (since sign(residual) is -1 and -delta*(-1)=delta)
        return -np.where(np.abs(residual) <= self.delta, residual, 
                         np.where(residual > 0, self.alpha * self.delta * np.sign(residual),
                                  self.delta * np.sign(residual)))
    
    def hessian_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        # Hessian is 1 in the quadratic region and 0 in the linear region.
        return np.where(np.abs(residual) <= self.delta, 1.0, 0.0)


class SmoothQuantileLoss(LossFunction):
    def __init__(self, tau=0.9, epsilon=1e-3):  # Smooth pinball loss
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.sum(np.where(np.abs(residual) < self.epsilon, 
                               0.5 * residual**2 / self.epsilon, 
                               np.maximum(self.tau * residual, (self.tau - 1) * residual)))

    def gradient_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        # In the smooth region: derivative = -residual/epsilon.
        # Else, if residual > 0: derivative = -tau; if residual <= 0: derivative = -(tau-1)
        return np.where(np.abs(residual) < self.epsilon, -residual / self.epsilon, 
                        np.where(residual > 0, -self.tau, -(self.tau - 1)))
    
    def hessian_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        # In the smooth region: second derivative is 1/epsilon; otherwise 0.
        return np.where(np.abs(residual) < self.epsilon, 1.0 / self.epsilon, 0.0)
