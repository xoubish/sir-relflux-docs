import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, interp1d

def interpolator2D(_x, _y, _z, grid=False):
    x = _x.copy()
    y = _x.copy()
    z = _z.copy()
    if grid:
        (x, y) = np.meshgrid(x, y)
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        z = z.reshape(z.size)
    points = np.vstack([x, y]).transpose()
    itp = CloughTocher2DInterpolator(points, z)
    return itp


#class Fit2D_NP:
class Fit2DNP:
    """ 2D profile fitting (non-parametric) """
    def grid(self):
        return np.linspace(-1., 1., self.npar+1)

    def midpoints(self):
        g = self.grid()
        x = (g[0:-1] + g[1:]) / 2.
        return x

    def __init__(self, npar, meas_x, meas_y):
        self.npar = npar
        assert meas_x.size == meas_y.size
        self.meas_x = meas_x.copy()
        self.meas_y = meas_y.copy()
        self.param_val = np.zeros((npar+1, npar+1))
        self.param_unc = np.zeros((npar+1, npar+1))
        self.bkp_val = self.param_val.copy()
        self.bkp_unc = self.param_unc.copy()
        mapx = np.digitize(self.meas_x, self.midpoints())
        mapy = np.digitize(self.meas_y, self.midpoints())
        self.indexmap = []
        for ix in range(0, npar+1):
            for iy in range(0, npar+1):
                j = np.where((mapx == ix)  &
                             (mapy == iy))[0]
                self.indexmap.append(j)

    def restore(self):
        self.param_val[:] = self.bkp_val
        self.param_unc[:] = self.bkp_unc

    def fit(self, meas, uncert):
        assert meas.ndim == 1
        assert meas.size == self.meas_x.size
        self.bkp_val[:] = self.param_val
        self.bkp_unc[:] = self.param_unc
        i = 0
        for ix in range(0, self.npar+1):
            for iy in range(0, self.npar+1):
                j = self.indexmap[i];  i += 1
                if len(j) > 0:
                    self.param_val[ix, iy] = np.mean(meas[j])
                    self.param_unc[ix, iy] = np.mean(uncert[j]) / np.sqrt(len(j))

    def extrapolate(self, x, y):
        assert np.isscalar(x)
        assert np.isscalar(y)
        assert ((np.abs(x) < 1)  or  (np.abs(y) < 1))
        if np.abs(x) < 1:
            mx = np.repeat(x, self.npar+1)
            my = self.grid()
            dd = self.predict_at(mx, my)
            return float(interp1d(my, dd, fill_value='extrapolate')(y))
        if np.abs(y) < 1:
            mx = self.grid()
            my = np.repeat(y, self.npar+1)
            dd = self.predict_at(mx, my)
            return float(interp1d(mx, dd, fill_value='extrapolate')(x))
        raise ValueError('Unexpected value for inputs (x=' + str(x) + ', y=' + str(y) + ').')

    def predict_at(self, x, y):
        itp = interpolator2D(self.grid(), self.grid(), self.param_val.transpose(), grid=True)
        return itp(x, y)

    def predict(self):
        return self.predict_at(self.meas_x, self.meas_y)

    def uncert_at(self, x, y):
        itp = interpolator2D(self.grid(), self.grid(), self.param_unc.transpose(), grid=True)
        return itp(x, y)

    def uncert(self):
        return self.uncert_at(self.meas_x, self.meas_y)

    def addGlobalOffset(self, offset):
        self.param_val[:] += offset


#class Fit2D_Cheb(Fit2D_NP):
class Fit2DCheb(Fit2DNP):
    """ 2D profile fitting based on Chebyshev Polynomials """
    def __init__(self, npar, meas_x, meas_y):
#        Fit2D_NP.__init__(self, npar, meas_x, meas_y)
        Fit2DNP.__init__(self, npar, meas_x, meas_y)
        self.itp = np.polynomial.chebyshev.chebvander2d(meas_x, meas_y, [npar, npar])

    def predict(self):
        return np.matmul(self.itp, self.param_val.reshape(self.param_val.size))

    def predict_at(self, x, y):
        itp = np.polynomial.chebyshev.chebvander2d(x, y, [self.npar, self.npar])
        return np.matmul(itp, self.param_val.reshape(self.param_val.size))

    def fit(self, meas, uncert):
#        Fit2D_NP.fit(self, meas, uncert)
        Fit2DNP.fit(self, meas, uncert)
        res = np.linalg.lstsq(self.itp, meas, rcond=-1.)
        self.param_val = res[0].reshape((self.npar+1, self.npar+1))
        normRank = res[2] / ((self.npar+1)**2)
        if normRank != 1.:
            print("Warning: normalized rank = ", normRank)

    def addGlobalOffset(self, offset):
        self.param_val[0,0] += offset
