"""
fe_axisym.py -- linear-elastic axisymmetric finite elements with 9-node
Lagrange quadrilaterals (Q9), for layered pavements under a circular load.

Units: N, mm, MPa.  Coordinates: r (radial, >= 0), z (depth, positive down).
Strain vector order: [eps_r, eps_z, eps_theta, gamma_rz].

The mesh is structured: element boundaries are given as two sorted arrays of
radial and depth coordinates, so every element is a rectangle and the
isoparametric map is diagonal. Layer interfaces and the load edge must be
element boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_G = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_W = np.array([5.0, 8.0, 5.0]) / 9.0


def _n1(x):
    """1-D quadratic shape functions at local nodes -1, 0, +1."""
    x = np.asarray(x, dtype=float)
    return np.stack([0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)], axis=-1)


def _dn1(x):
    x = np.asarray(x, dtype=float)
    return np.stack([x - 0.5, -2.0 * x, x + 0.5], axis=-1)


def elasticity(E, nu):
    """Axisymmetric isotropic elasticity matrix, order (r, z, theta, rz)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    d = np.zeros((4, 4))
    d[:3, :3] = lam
    d[[0, 1, 2], [0, 1, 2]] = lam + 2.0 * mu
    d[3, 3] = mu
    return d


def geometric_edges(x0, x1, n, h0=None):
    """n elements from x0 to x1; geometric growth starting at size h0 (uniform if None)."""
    L = x1 - x0
    if h0 is None or abs(h0 * n - L) < 1e-9:
        return np.linspace(x0, x1, n + 1)
    lo, hi = 1.0 + 1e-12, 50.0
    if h0 * n > L:
        lo, hi = 1e-3, 1.0 - 1e-12
    for _ in range(200):
        q = 0.5 * (lo + hi)
        s = h0 * (q ** n - 1.0) / (q - 1.0)
        if s > L:
            hi = q
        else:
            lo = q
    h = h0 * q ** np.arange(n)
    h *= L / h.sum()
    return np.concatenate([[x0], x0 + np.cumsum(h)])


@dataclass
class Layer:
    name: str
    z_top: float
    z_bot: float
    nu: float


@dataclass
class Mesh:
    r_edges: np.ndarray
    z_edges: np.ndarray
    layers: list
    nr: int = field(init=False)
    nz: int = field(init=False)

    def __post_init__(self):
        self.r_edges = np.asarray(self.r_edges, float)
        self.z_edges = np.asarray(self.z_edges, float)
        self.nr, self.nz = len(self.r_edges) - 1, len(self.z_edges) - 1
        rn = np.empty(2 * self.nr + 1)
        rn[0::2] = self.r_edges
        rn[1::2] = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        zn = np.empty(2 * self.nz + 1)
        zn[0::2] = self.z_edges
        zn[1::2] = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        self.r_nodes, self.z_nodes = rn, zn
        self.nnr, self.nnz = len(rn), len(zn)
        self.n_nodes = self.nnr * self.nnz
        self.n_dof = 2 * self.n_nodes
        zc = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        layer_j = np.array([self._layer_of(z) for z in zc])
        # connectivity: element (i, j) -> 9 node ids, local order (a radial, b depth)
        conn = np.empty((self.nr * self.nz, 9), dtype=np.int64)
        e = 0
        for j in range(self.nz):
            for i in range(self.nr):
                k = 0
                for b in range(3):
                    for a in range(3):
                        conn[e, k] = (2 * j + b) * self.nnr + (2 * i + a)
                        k += 1
                e += 1
        self.conn = conn
        self.elem_i = np.tile(np.arange(self.nr), self.nz)
        self.elem_j = np.repeat(np.arange(self.nz), self.nr)
        self.elem_layer = layer_j[self.elem_j]

    def _layer_of(self, z):
        for k, L in enumerate(self.layers):
            if L.z_top <= z <= L.z_bot:
                return k
        raise ValueError(f"depth {z} outside layers")

    @property
    def n_elem(self):
        return self.nr * self.nz

    def refined(self, factor=2):
        def split(e):
            out = [e[0]]
            for a, b in zip(e[:-1], e[1:]):
                out.extend(list(np.linspace(a, b, factor + 1)[1:]))
            return np.array(out)
        return Mesh(split(self.r_edges), split(self.z_edges), self.layers)


def _local_B(xi, eta, hr, hz, r):
    """B matrix (4 x 18) at local point for a rectangular Q9 element; dofs (ur, uz) per node."""
    Nx, Ny = _n1(xi), _n1(eta)
    dNx, dNy = _dn1(xi), _dn1(eta)
    N = np.outer(Ny, Nx).ravel()                    # node order: b outer, a inner
    dNdr = np.outer(Ny, dNx).ravel() * (2.0 / hr)
    dNdz = np.outer(dNy, Nx).ravel() * (2.0 / hz)
    B = np.zeros((4, 18))
    B[0, 0::2] = dNdr
    B[1, 1::2] = dNdz
    B[2, 0::2] = N / r
    B[3, 0::2] = dNdz
    B[3, 1::2] = dNdr
    return B, N


class AxisymModel:
    """Assembles unit-modulus stiffness once; solves for any set of layer moduli."""

    def __init__(self, mesh: Mesh, nu_override: dict | None = None):
        self.mesh = mesh
        nus = [L.nu for L in mesh.layers]
        if nu_override:
            for k, L in enumerate(mesh.layers):
                if L.name in nu_override:
                    nus[k] = nu_override[L.name]
        self.nus = nus
        m = mesh
        ke = np.zeros((m.n_elem, 18, 18))
        for e in range(m.n_elem):
            i, j = m.elem_i[e], m.elem_j[e]
            r0, r1 = m.r_edges[i], m.r_edges[i + 1]
            z0, z1 = m.z_edges[j], m.z_edges[j + 1]
            hr, hz = r1 - r0, z1 - z0
            D = elasticity(1.0, nus[m.elem_layer[e]])
            k = np.zeros((18, 18))
            for gx, wx in zip(_G, _W):
                r = r0 + 0.5 * (gx + 1.0) * hr
                for gy, wy in zip(_G, _W):
                    B, _ = _local_B(gx, gy, hr, hz, r)
                    k += B.T @ D @ B * (2.0 * np.pi * r * 0.25 * hr * hz * wx * wy)
            ke[e] = k
        self.ke = ke
        dofs = np.empty((m.n_elem, 18), dtype=np.int64)
        dofs[:, 0::2] = 2 * m.conn
        dofs[:, 1::2] = 2 * m.conn + 1
        self.dofs = dofs
        self.rows = np.repeat(dofs, 18, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 18)).ravel()
        # boundary conditions
        nid = np.arange(m.n_nodes).reshape(m.nnz, m.nnr)
        fixed = set((2 * nid[:, 0]).tolist())                 # u_r = 0 on axis
        fixed |= set((2 * nid[:, -1]).tolist())               # u_r = 0 on outer boundary
        fixed |= set((2 * nid[-1, :]).tolist())               # bottom: u_r = 0
        fixed |= set((2 * nid[-1, :] + 1).tolist())           # bottom: u_z = 0
        self.fixed = np.array(sorted(fixed))
        self.free = np.setdiff1d(np.arange(m.n_dof), self.fixed)

    def load_vector(self, p, a):
        """Consistent nodal forces for uniform vertical pressure p (MPa) on 0 <= r <= a at z = 0."""
        m = self.mesh
        if not np.any(np.isclose(m.r_edges, a)):
            raise ValueError("load radius must coincide with an element edge")
        f = np.zeros(m.n_dof)
        for i in range(m.nr):
            r0, r1 = m.r_edges[i], m.r_edges[i + 1]
            if r1 > a + 1e-9:
                break
            hr = r1 - r0
            fe = np.zeros(3)
            for gx, wx in zip(_G, _W):
                r = r0 + 0.5 * (gx + 1.0) * hr
                fe += _n1(gx) * p * 2.0 * np.pi * r * 0.5 * hr * wx
            for a_loc in range(3):
                node = 2 * i + a_loc                          # top row of nodes (z index 0)
                f[2 * node + 1] += fe[a_loc]
        return f

    def solve(self, layer_E, p=0.70, a=106.5):
        """layer_E: list of moduli (MPa), one per mesh layer. Returns a Solution."""
        m = self.mesh
        Ee = np.asarray(layer_E, float)[m.elem_layer]
        data = (self.ke * Ee[:, None, None]).ravel()
        K = sp.csr_matrix((data, (self.rows, self.cols)), shape=(m.n_dof, m.n_dof))
        f = self.load_vector(p, a)
        fr = self.free
        u = np.zeros(m.n_dof)
        u[fr] = spla.spsolve(K[fr][:, fr].tocsc(), f[fr])
        return Solution(self, u, np.asarray(layer_E, float))


class Solution:
    def __init__(self, model: AxisymModel, u, layer_E):
        self.model, self.u, self.layer_E = model, u, layer_E

    def _locate(self, r, z, side):
        m = self.model.mesh
        i = int(np.clip(np.searchsorted(m.r_edges, r, side="right") - 1, 0, m.nr - 1))
        if side == "above":
            j = int(np.clip(np.searchsorted(m.z_edges, z, side="left") - 1, 0, m.nz - 1))
        else:
            j = int(np.clip(np.searchsorted(m.z_edges, z, side="right") - 1, 0, m.nz - 1))
        return i, j

    def strain_stress(self, r, z, side="below"):
        """Strain and stress at (r, z). side='above' takes the element above an interface."""
        m = self.model.mesh
        r = max(r, 1e-6)
        i, j = self._locate(r, z, side)
        e = j * m.nr + i
        r0, r1 = m.r_edges[i], m.r_edges[i + 1]
        z0, z1 = m.z_edges[j], m.z_edges[j + 1]
        xi = 2.0 * (r - r0) / (r1 - r0) - 1.0
        eta = 2.0 * (z - z0) / (z1 - z0) - 1.0
        B, _ = _local_B(xi, eta, r1 - r0, z1 - z0, r)
        eps = B @ self.u[self.model.dofs[e]]
        lay = m.elem_layer[e]
        sig = elasticity(self.layer_E[lay], self.model.nus[lay]) @ eps
        return eps, sig

    def surface_uz(self, r):
        m = self.model.mesh
        i = int(np.clip(np.searchsorted(m.r_edges, r, side="right") - 1, 0, m.nr - 1))
        r0, r1 = m.r_edges[i], m.r_edges[i + 1]
        xi = 2.0 * (r - r0) / (r1 - r0) - 1.0
        nodes = [2 * i, 2 * i + 1, 2 * i + 2]
        return float(_n1(xi) @ self.u[[2 * n + 1 for n in nodes]])


def boussinesq_sigma_z_axis(p, a, z):
    """Vertical stress on the axis under a uniform circular load on a half-space."""
    return p * (1.0 - z ** 3 / (a ** 2 + z ** 2) ** 1.5)


def boussinesq_w0(p, a, E, nu):
    """Surface deflection at the centre of a flexible uniform circular load."""
    return 2.0 * (1.0 - nu ** 2) * p * a / E
