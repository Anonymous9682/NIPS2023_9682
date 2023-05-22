import torch
from gpytorch.kernels import RBFKernel, PolynomialKernel

class Gaussian_kernel():
    def __init__(self, gamma=None) -> None:
        if gamma is None: gamma = 1.0
        self.gamma = gamma
        self.kernel = RBFKernel()
        RBFKernel.lengthscale = torch.sqrt(0.5/torch.tensor(self.gamma)).reshape(1, 1).to('cuda')

    def __call__(self, x1, x2=None) -> torch.tensor:
        kernel_matrix = self.kernel(x1, x2)
        return kernel_matrix.evaluate()

    def name(self):
        return 'gauss{}'.format(self.gamma) 
    
class linear():
    def __init__(self):
        self.kernel = PolynomialKernel(power=1)
        self.kernel.offset = torch.tensor(0).double()
        self.kernel = self.kernel.double().to('cuda')

    def __call__(self, x1, x2=None):
        kernel_matrix = self.kernel(x1, x2)
        return kernel_matrix.evaluate()

    def name(self):
        return 'linear'
    
class poly_kernel():
    def __init__(self,d,c=None) -> None:
        if not c: 
            c = 0
        self.kernel = PolynomialKernel(power=d)
        self.kernel.offset = torch.tensor(c).double()
        self.kernel = self.kernel.double().to('cuda')
    
    def __call__(self, x1, x2=None) -> torch.tensor:
        kernel_matrix = self.kernel(x1, x2)
        return kernel_matrix.evaluate()

    def name(self):
        return 'poly{}'.format(self.d)
    