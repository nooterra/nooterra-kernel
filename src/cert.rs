//! Exact-rational 3D simplicial certificate data structures (`no_std`).

use alloc::vec::Vec;

use crate::sha256::Sha256;

/// Index lookup for the 10 components of a totally-symmetric 3rd-order tensor `H_{ijk}` in 3D:
/// 0: 000, 1: 111, 2: 222, 3: 001, 4: 002, 5: 011, 6: 112, 7: 022, 8: 122, 9: 012.
pub const H_IDX: [[[usize; 3]; 3]; 3] = [
    [[0, 3, 4], [3, 5, 9], [4, 9, 7]],
    [[3, 5, 9], [5, 1, 6], [9, 6, 8]],
    [[4, 9, 7], [9, 6, 8], [7, 8, 2]],
];

/// Index lookup for the 6 components of a symmetric 3x3 tensor `g_{ij}`:
/// 0: 00, 1: 11, 2: 22, 3: 01, 4: 02, 5: 12.
pub const G_IDX: [[usize; 3]; 3] = [
    [0, 3, 4],
    [3, 1, 5],
    [4, 5, 2],
];

/// Engineering-strain pair `(j, k)` for the 6 symmetric strain basis entries `e = (e00, e11, e22, 2e01, 2e02, 2e12)`.
pub const STRAIN_PAIR: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

/// Exact rational corner of a polytopic material / frequency uncertainty box:
/// `mu = mu_num / mat_den`, `lam = lam_num / mat_den`, `omega^2 * rho = omega2_rho_num / mat_den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialCorner {
    pub mu_num: i64,
    pub lam_num: i64,
    pub omega2_rho_num: i64,
}

/// Classification of an oriented boundary triangle `(v0, v1, v2)` of the 3D tetrahedral mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryFaceKind {
    /// Clamped zero-displacement face `u = 0` (`g` and `H` are free).
    Dirichlet,
    /// Traction boundary face with exact rational traction `t = traction_num / traction_den`
    /// (enforces `2 * g * n = -t` and `-H * n >= 0` at all 3 vertices).
    Neumann {
        traction_num: [i64; 3],
        traction_den: u64,
    },
    /// Compositional subdomain interface face (verified by `verify_composed` to match opposite
    /// neighbor subdomain with identical exact rational `(g, H)` traces).
    Interface { tag: u32 },
    /// 3D unilateral contact face with exact 3D circular Coulomb friction cone
    /// (`p_n >= 0` and `sqrt(tau_1^2 + tau_2^2) <= mu * p_n`, where `mu = mu_num / mu_den`)
    /// and optional initial geometric flatness tolerance gap `g_0 = gap_num / gap_den >= 0`.
    Contact {
        mu_num: i64,
        mu_den: u64,
        gap_num: i64,
        gap_den: u64,
    },
}

/// Boundary face assignment linking an oriented mesh boundary triangle to its physical BC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryFaceSpec {
    pub verts: [u32; 3],
    pub kind: BoundaryFaceKind,
}

/// Exact-rational 3D tetrahedral elasticity specification with polytopic material/frequency corners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec3D {
    pub coord_den: u64,
    pub verts: Vec<[i64; 3]>,
    pub tets: Vec<[u32; 4]>,
    pub mat_den: u64,
    pub material_corners: Vec<MaterialCorner>,
    pub boundary_faces: Vec<BoundaryFaceSpec>,
}

impl Spec3D {
    /// Deterministic SHA-256 hash binding this exact specification and kernel version.
    pub fn spec_hash(&self, kernel_id: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(kernel_id.as_bytes());
        h.update_u64(self.coord_den);
        h.update_u64(self.verts.len() as u64);
        for v in &self.verts {
            for &c in v {
                h.update_i64(c);
            }
        }
        h.update_u64(self.tets.len() as u64);
        for t in &self.tets {
            for &idx in t {
                h.update_u64(idx as u64);
            }
        }
        h.update_u64(self.mat_den);
        h.update_u64(self.material_corners.len() as u64);
        for mc in &self.material_corners {
            h.update_i64(mc.mu_num);
            h.update_i64(mc.lam_num);
            h.update_i64(mc.omega2_rho_num);
        }
        h.update_u64(self.boundary_faces.len() as u64);
        for bf in &self.boundary_faces {
            for &idx in &bf.verts {
                h.update_u64(idx as u64);
            }
            match bf.kind {
                BoundaryFaceKind::Dirichlet => h.update(&[0]),
                BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                    h.update(&[1]);
                    for &c in &traction_num {
                        h.update_i64(c);
                    }
                    h.update_u64(traction_den);
                }
                BoundaryFaceKind::Interface { tag } => {
                    h.update(&[2]);
                    h.update_u64(tag as u64);
                }
                BoundaryFaceKind::Contact { mu_num, mu_den, gap_num, gap_den } => {
                    h.update(&[3]);
                    h.update_i64(mu_num);
                    h.update_u64(mu_den);
                    h.update_i64(gap_num);
                    h.update_u64(gap_den);
                }
            }
        }
        h.finalize()
    }
}

/// Exact-rational dual flux witness for a 3D simplicial certificate (`10x10` strain-Riccati form).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness3D {
    /// Positive common denominator for all nodal fluxes `g`, `H`, and corner scalars `c`.
    pub flux_den: u64,
    /// Per-node symmetric stress flux `g` (6 components: `00, 11, 22, 01, 02, 12`).
    pub g_nodes: Vec<[i64; 6]>,
    /// Per-node totally-symmetric Riccati flux `H` (10 components, see `H_IDX`).
    pub h_nodes: Vec<[i64; 10]>,
    /// Per-tet-corner lower bound scalars `c` (`4 * tets.len()` entries).
    pub c_corners: Vec<i64>,
}

/// Complete 3D simplicial certificate pairing a specification with its dual flux witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimplicialCert3D {
    pub spec: Spec3D,
    pub witness: Witness3D,
}
