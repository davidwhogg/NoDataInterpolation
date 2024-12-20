import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator, Diagonal
from typing import Optional

class FrizzleOperator(LinearOperator):

    def __init__(self, x: npt.ArrayLike, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain (0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        if x.dtype == np.float64:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float64, np.complex128)
        else:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float32, np.complex64)
        super().__init__(dtype=x.dtype, shape=(len(x), n_modes))
        self.explicit = False
        self.finufft_kwds = dict(dtype=self.DTYPE_COMPLEX.__name__, n_modes_or_dim=(n_modes, ), modeord=0, **kwargs)
        self._plan_matvec = finufft.Plan(2, **self.finufft_kwds)
        self._plan_rmatvec = finufft.Plan(1, **self.finufft_kwds)
        self._plan_matvec.setpts(x)
        self._plan_rmatvec.setpts(x)
        self._hx = n_modes // 2 
        return None
    
    def _pre_process_matvec(self, c):
        return np.hstack([-1j * c[:self._hx], c[self._hx:]], dtype=self.DTYPE_COMPLEX)

    def _post_process_rmatvec(self, f):
        return np.hstack([-f[:self._hx].imag, f[self._hx:].real], dtype=self.DTYPE_REAL)

    def _matvec(self, c):
        return self._plan_matvec.execute(self._pre_process_matvec(c)).real

    def _rmatvec(self, f):
        return self._post_process_rmatvec(self._plan_rmatvec.execute(f.astype(self.DTYPE_COMPLEX)))
