//! Locking-Free 3D Conforming Hu-Man-Zhang (`HMZ1`) $H(\mathrm{div}; \mathbb{S})$ Proof Checker
//! (`no_std`, `#![forbid(unsafe_code)]`, 100% PURE EXACT INTEGER ARITHMETIC — ZERO FLOATS!).
//!
//! Eliminates 3D P1 tetrahedral symmetric-stress locking by representing symmetric stress fields
//! on structured 3D hexahedral grids `(nx, ny, nz)` with:
//!   - Normal stresses `fx` (`sigma_xx`), `fy` (`sigma_yy`), `fz` (`sigma_zz`) stored on axis faces
//!     (quadratic along their axis with exact equilibrium bubbles `q_x, q_y, q_z`),
//!   - Shear stresses `ex` (`sigma_yz`), `ey` (`sigma_xz`), `ez` (`sigma_xy`) stored on grid edges
//!     (bilinear across the two orthogonal axes).
//!
//! Independently verifies:
//!   1. Exact pointwise equilibrium `div(sigma) === 0` at EVERY point of EVERY 3D cell (`0/1` exact).
//!   2. Exact rational Neumann boundary tractions on all classified Neumann faces and edges.
//!   3. Exact closed-form Legendre orthogonal rational integral `C(E) = int_Omega sigma : C(E)^-1 : sigma dV`
//!      across EVERY corner of the polytopic material uncertainty box `E in [E_min, E_max]`
//!      with ZERO quadrature error and ZERO floating-point rounding!
//!   4. Exact 1-to-1 compositional interface contract matching across subdomains (`verify_composed_hmz`).

use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};
use crate::cert::{BoundaryFaceKind, MaterialCorner};
use crate::sha256::Sha256;

/// Kernel identity baked into every HMZ1 SHA-256 specification hash.
pub const HMZ_KERNEL_ID: &str = "nooterra-kernel/2.2.0 hex3d-hmz1-contact-yield-exact";

/// Exact-rational 3D box specification for locking-free HMZ1 certification (with optional
/// 3D unilateral circular-friction contact faces & exact pointwise 3D von Mises yield limit).
/// Cell dimensions: `dx = dx_num / grid_den`, `dy = dy_num / grid_den`, `dz = dz_num / grid_den`.
/// Box sides `face_kinds[0..6]`: `0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HmzSpec3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub grid_den: u64,
    pub dx_num: u64,
    pub dy_num: u64,
    pub dz_num: u64,
    pub mat_den: u64,
    pub material_corners: Vec<MaterialCorner>,
    pub face_kinds: [BoundaryFaceKind; 6],
    /// Optional exact rational von Mises yield strength `sigma_Y = num / den` (`num > 0, den > 0`).
    /// When `Some`, every point of every 3D cube is certified to satisfy `sigma_vM(x) <= sigma_Y`!
    pub yield_limit: Option<(i64, u64)>,
}

impl HmzSpec3D {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(HMZ_KERNEL_ID.as_bytes());
        h.update_u64(self.nx as u64);
        h.update_u64(self.ny as u64);
        h.update_u64(self.nz as u64);
        h.update_u64(self.grid_den);
        h.update_u64(self.dx_num);
        h.update_u64(self.dy_num);
        h.update_u64(self.dz_num);
        h.update_u64(self.mat_den);
        h.update_u64(self.material_corners.len() as u64);
        for mc in &self.material_corners {
            h.update_i64(mc.mu_num);
            h.update_i64(mc.lam_num);
            h.update_i64(mc.omega2_rho_num);
        }
        for fk in &self.face_kinds {
            match *fk {
                BoundaryFaceKind::Dirichlet => h.update(&[0]),
                BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                    h.update(&[1]);
                    for c in traction_num {
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
        match self.yield_limit {
            None => h.update(&[0]),
            Some((yn, yd)) => {
                h.update(&[1]);
                h.update_i64(yn);
                h.update_u64(yd);
            }
        }
        h.finalize()
    }
}

/// Exact-integer HMZ1 symmetric stress witness (physical stress = `int_val / flux_den`).
/// Indexing conventions:
///   - `fx[(i * ny + j) * nz + k]`           : `i in 0..=nx, j in 0..ny, k in 0..nz` (`sigma_xx`)
///   - `fy[(i * (ny + 1) + j) * nz + k]`     : `i in 0..nx, j in 0..=ny, k in 0..nz` (`sigma_yy`)
///   - `fz[(i * ny + j) * (nz + 1) + k]`     : `i in 0..nx, j in 0..ny, k in 0..=nz` (`sigma_zz`)
///   - `ex[(i * (ny + 1) + j) * (nz + 1) + k]`: `i in 0..nx, j in 0..=ny, k in 0..=nz` (`sigma_yz`)
///   - `ey[(i * ny + j) * (nz + 1) + k]`     : `i in 0..=nx, j in 0..ny, k in 0..=nz` (`sigma_xz`)
///   - `ez[(i * (ny + 1) + j) * nz + k]`     : `i in 0..=nx, j in 0..=ny, k in 0..nz` (`sigma_xy`)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HmzWitness3D {
    pub flux_den: u64,
    pub fx: Vec<i64>,
    pub fy: Vec<i64>,
    pub fz: Vec<i64>,
    pub ex: Vec<i64>,
    pub ey: Vec<i64>,
    pub ez: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HmzCert3D {
    pub spec: HmzSpec3D,
    pub witness: HmzWitness3D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HmzRejection {
    InvalidGrid,
    InvalidDenominator,
    EmptyUncertaintyBox,
    InvalidMaterialParams,
    InvalidContactParams,
    InvalidYieldLimit,
    WitnessSizeMismatch,
    NeumannNormalMismatch { side: u8, idx: usize },
    NeumannShearMismatch { side: u8, idx: usize },
    ContactTensionViolation { side: u8, j: usize, k: usize },
    ContactCircularFrictionViolation { side: u8, j: usize, k: usize },
    UnsupportedContactSide { side: u8 },
    EquilibriumViolation { i: usize, j: usize, k: usize, axis: u8 },
    YieldLimitExceeded { i: usize, j: usize, k: usize },
    InterfaceGeometryMismatch,
    InterfaceTagMismatch,
    InterfaceTractionMismatch { kind: u8, idx: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HmzProofReport {
    pub spec_hash: [u8; 32],
    pub num_cells: usize,
    pub num_mat_corners: usize,
    /// Exact rational compliance upper bound `C_upper(mc)` (including any contact gap work `2 int p_n g_0`)
    /// for EVERY material corner `mc`.
    pub compliance_per_corner: Vec<BigRat>,
    /// Worst-case exact rational compliance upper bound across the entire uncertainty polytope!
    pub worst_compliance_upper: BigRat,
    /// Exact rational global upper bound on pointwise squared von Mises stress `max_{x in Omega} sigma_vM(x)^2`
    /// (zero quadrature error via 3-point Bernstein normal intervals + 8-corner exact BigInt check!).
    pub max_von_mises_sq: BigRat,
    /// Exact rational contact gap work `2 int_{Gamma_C} p_n g_0 dA` (`0` if `g_0 = 0` or no contact).
    pub contact_gap_compliance: BigRat,
    /// Number of 2D contact squares verified against the exact 3D circular Coulomb friction cone.
    pub contact_faces_verified: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HmzComposedReport {
    pub report_a: HmzProofReport,
    pub report_b: HmzProofReport,
    pub matched_interface_dofs: usize,
    pub total_worst_compliance_upper: BigRat,
}

#[inline]
pub fn idx_fx(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * ny + j) * nz + k
}
#[inline]
pub fn idx_fy(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * (ny + 1) + j) * nz + k
}
#[inline]
pub fn idx_fz(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * ny + j) * (nz + 1) + k
}
#[inline]
pub fn idx_ex(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * (ny + 1) + j) * (nz + 1) + k
}
#[inline]
pub fn idx_ey(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * ny + j) * (nz + 1) + k
}
#[inline]
pub fn idx_ez(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * (ny + 1) + j) * nz + k
}

/// Exact integer verification of `sgn * stress_int / flux_den == traction_num / traction_den`.
#[inline]
fn check_exact_bc(stress_int: i64, flux_den: u64, sgn: i128, t_num: i64, t_den: u64) -> bool {
    let lhs = sgn * (stress_int as i128) * (t_den as i128);
    let rhs = (t_num as i128) * (flux_den as i128);
    lhs == rhs
}

/// Zero-trust pure-exact-integer verifier for a 3D HMZ1 certificate.
/// Zero floating-point operations anywhere!
pub fn verify_hmz_cert(cert: &HmzCert3D) -> Result<HmzProofReport, HmzRejection> {
    let s = &cert.spec;
    let w = &cert.witness;
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    if nx == 0 || ny == 0 || nz == 0 || s.dx_num == 0 || s.dy_num == 0 || s.dz_num == 0 {
        return Err(HmzRejection::InvalidGrid);
    }
    if s.grid_den == 0 || s.mat_den == 0 || w.flux_den == 0 {
        return Err(HmzRejection::InvalidDenominator);
    }
    if s.material_corners.is_empty() {
        return Err(HmzRejection::EmptyUncertaintyBox);
    }
    for mc in &s.material_corners {
        if mc.mu_num <= 0 || (2i128 * (mc.mu_num as i128) + 3i128 * (mc.lam_num as i128)) <= 0 {
            return Err(HmzRejection::InvalidMaterialParams);
        }
    }
    if let Some((y_num, y_den)) = s.yield_limit {
        if y_num <= 0 || y_den == 0 {
            return Err(HmzRejection::InvalidYieldLimit);
        }
    }
    if w.fx.len() != (nx + 1) * ny * nz
        || w.fy.len() != nx * (ny + 1) * nz
        || w.fz.len() != nx * ny * (nz + 1)
        || w.ex.len() != nx * (ny + 1) * (nz + 1)
        || w.ey.len() != (nx + 1) * ny * (nz + 1)
        || w.ez.len() != (nx + 1) * (ny + 1) * nz
    {
        return Err(HmzRejection::WitnessSizeMismatch);
    }

    // 1. Exact rational Neumann AND 3D unilateral circular-friction Contact checks on all 6 sides!
    let mut contact_gap_compliance = BigRat::zero();
    let mut contact_faces_verified = 0usize;
    let gd_u = BigUint::from_u64(s.grid_den);
    let gd2_u = gd_u.mul_u(&gd_u);
    let fd_u = BigUint::from_u64(w.flux_den);

    // Side 0: x_min (n = -x), Side 1: x_max (n = +x)
    for (side, i_face, sgn) in [(0u8, 0usize, -1i128), (1u8, nx, 1i128)] {
        match s.face_kinds[side as usize] {
            BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                if traction_den == 0 {
                    return Err(HmzRejection::InvalidDenominator);
                }
                for j in 0..ny {
                    for k in 0..nz {
                        let id = idx_fx(ny, nz, i_face, j, k);
                        if !check_exact_bc(w.fx[id], w.flux_den, sgn, traction_num[0], traction_den) {
                            return Err(HmzRejection::NeumannNormalMismatch { side, idx: id });
                        }
                    }
                }
                for j in 0..=ny {
                    for k in 0..nz {
                        let id = idx_ez(ny, nz, i_face, j, k);
                        if !check_exact_bc(w.ez[id], w.flux_den, sgn, traction_num[1], traction_den) {
                            return Err(HmzRejection::NeumannShearMismatch { side, idx: id });
                        }
                    }
                }
                for j in 0..ny {
                    for k in 0..=nz {
                        let id = idx_ey(ny, nz, i_face, j, k);
                        if !check_exact_bc(w.ey[id], w.flux_den, sgn, traction_num[2], traction_den) {
                            return Err(HmzRejection::NeumannShearMismatch { side, idx: id });
                        }
                    }
                }
            }
            BoundaryFaceKind::Contact { mu_num, mu_den, gap_num, gap_den } => {
                if mu_num < 0 || mu_den == 0 || gap_den == 0 {
                    return Err(HmzRejection::InvalidContactParams);
                }
                // On BOTH x_min (n=-x, sigma_n = sigma_xx) and x_max (n=+x, sigma_n = sigma_xx),
                // compressive contact pressure is p_n = -sigma_xx = -fx >= 0!
                // On each 2D contact square (j, k), sigma_xy is linear in eta (ez0..ez1) and
                // sigma_xz is linear in zeta (ey0..ey1), so the exact continuous maximum of
                // sigma_xy^2 + sigma_xz^2 over the square is max(ez0^2, ez1^2) + max(ey0^2, ey1^2)!
                let mu_num_u = BigUint::from_u64(mu_num as u64);
                let mu_num2 = mu_num_u.mul_u(&mu_num_u);
                let mu_den_u = BigUint::from_u64(mu_den);
                let mu_den2 = mu_den_u.mul_u(&mu_den_u);
                let mut sum_pn = BigUint::zero();

                for j in 0..ny {
                    for k in 0..nz {
                        let fx_v = w.fx[idx_fx(ny, nz, i_face, j, k)] as i128;
                        let pn = -fx_v;
                        if pn < 0 {
                            return Err(HmzRejection::ContactTensionViolation { side, j, k });
                        }
                        let pn_u = BigUint::from_u128(pn as u128);
                        sum_pn = sum_pn.add_u(&pn_u);

                        let ez0 = (w.ez[idx_ez(ny, nz, i_face, j, k)] as i128).unsigned_abs();
                        let ez1 = (w.ez[idx_ez(ny, nz, i_face, j + 1, k)] as i128).unsigned_abs();
                        let ey0 = (w.ey[idx_ey(ny, nz, i_face, j, k)] as i128).unsigned_abs();
                        let ey1 = (w.ey[idx_ey(ny, nz, i_face, j, k + 1)] as i128).unsigned_abs();
                        let max_ez = BigUint::from_u128(if ez0 >= ez1 { ez0 } else { ez1 });
                        let max_ey = BigUint::from_u128(if ey0 >= ey1 { ey0 } else { ey1 });
                        let tau_sq = max_ez.mul_u(&max_ez).add_u(&max_ey.mul_u(&max_ey));
                        let lhs = tau_sq.mul_u(&mu_den2);
                        let rhs = pn_u.mul_u(&pn_u).mul_u(&mu_num2);
                        if lhs > rhs {
                            return Err(HmzRejection::ContactCircularFrictionViolation { side, j, k });
                        }
                        contact_faces_verified += 1;
                    }
                }
                if gap_num != 0 && !sum_pn.is_zero() {
                    // Exact gap compliance contribution: 2 * (dy_num*dz_num/grid_den^2) * (gap_num/gap_den) * (sum_pn/flux_den)
                    let area_2 = BigUint::from_u128(2u128 * (s.dy_num as u128) * (s.dz_num as u128));
                    let gap_mag = BigUint::from_u128((gap_num as i128).unsigned_abs());
                    let num = BigInt {
                        neg: gap_num < 0,
                        mag: area_2.mul_u(&gap_mag).mul_u(&sum_pn),
                    };
                    let den = gd2_u.mul_u(&BigUint::from_u64(gap_den)).mul_u(&fd_u);
                    contact_gap_compliance = contact_gap_compliance.add(&BigRat::new(num, den));
                }
            }
            BoundaryFaceKind::Dirichlet | BoundaryFaceKind::Interface { .. } => {}
        }
    }
    // Side 2: y_min (n = -y), Side 3: y_max (n = +y)
    for (side, j_face, sgn) in [(2u8, 0usize, -1i128), (3u8, ny, 1i128)] {
        match s.face_kinds[side as usize] {
            BoundaryFaceKind::Contact { .. } => {
                return Err(HmzRejection::UnsupportedContactSide { side });
            }
            BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                if traction_den == 0 {
                    return Err(HmzRejection::InvalidDenominator);
                }
                for i in 0..nx {
                    for k in 0..nz {
                        let id = idx_fy(ny, nz, i, j_face, k);
                        if !check_exact_bc(w.fy[id], w.flux_den, sgn, traction_num[1], traction_den) {
                            return Err(HmzRejection::NeumannNormalMismatch { side, idx: id });
                        }
                    }
                }
                for i in 0..=nx {
                    for k in 0..nz {
                        let id = idx_ez(ny, nz, i, j_face, k);
                        if !check_exact_bc(w.ez[id], w.flux_den, sgn, traction_num[0], traction_den) {
                            return Err(HmzRejection::NeumannShearMismatch { side, idx: id });
                        }
                    }
                }
                for i in 0..nx {
                    for k in 0..=nz {
                        let id = idx_ex(ny, nz, i, j_face, k);
                        if !check_exact_bc(w.ex[id], w.flux_den, sgn, traction_num[2], traction_den) {
                            return Err(HmzRejection::NeumannShearMismatch { side, idx: id });
                        }
                    }
                }
            }
            BoundaryFaceKind::Dirichlet | BoundaryFaceKind::Interface { .. } => {}
        }
    }
    // Side 4: z_min (n = -z), Side 5: z_max (n = +z)
    for (side, k_face, sgn) in [(4u8, 0usize, -1i128), (5u8, nz, 1i128)] {
        match s.face_kinds[side as usize] {
            BoundaryFaceKind::Contact { .. } => {
                return Err(HmzRejection::UnsupportedContactSide { side });
            }
            BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                if traction_den == 0 {
                    return Err(HmzRejection::InvalidDenominator);
                }
                for i in 0..nx {
                    for j in 0..ny {
                        let id = idx_fz(ny, nz, i, j, k_face);
                        if !check_exact_bc(w.fz[id], w.flux_den, sgn, traction_num[2], traction_den) {
                            return Err(HmzRejection::NeumannNormalMismatch { side, idx: id });
                        }
                    }
                }
                for i in 0..=nx {
                    for j in 0..ny {
                        let id = idx_ey(ny, nz, i, j, k_face);
                        if !check_exact_bc(w.ey[id], w.flux_den, sgn, traction_num[0], traction_den) {
                            return Err(HmzRejection::NeumannShearMismatch { side, idx: id });
                        }
                    }
                }
                for i in 0..nx {
                    for j in 0..=ny {
                        let id = idx_ex(ny, nz, i, j, k_face);
                        if !check_exact_bc(w.ex[id], w.flux_den, sgn, traction_num[1], traction_den) {
                            return Err(HmzRejection::NeumannShearMismatch { side, idx: id });
                        }
                    }
                }
            }
            BoundaryFaceKind::Dirichlet | BoundaryFaceKind::Interface { .. } => {}
        }
    }

    // 2. Exact pointwise equilibrium check (div(sigma) === 0 everywhere inside every cube!)
    //    AND exact integer accumulation of the 4 orthogonal Legendre quadratic energy sums!
    let dx = s.dx_num as i128;
    let dy = s.dy_num as i128;
    let dz = s.dz_num as i128;
    let dxy = dx * dy;
    let dxz = dx * dz;
    let dyz = dy * dz;
    let dxyz = dxy * dz;

    // All normal-stress orthogonal modes are scaled by `L = 24 * dxyz`:
    //   m_a * L = 12 * dxyz * (F_L + F_R)
    //   d_a * L = 12 * dxyz * (F_R - F_L)
    //   q_a * L = -3 * d_a * (d_a * d_c * twist_b + d_a * d_b * twist_c)
    //   (2/3) * q_a * L = -2 * d_a * (d_a * d_c * twist_b + d_a * d_b * twist_c)
    //   sbar_a * L = m_a * L - (2/3) * q_a * L
    // Shear orthogonal modes are scaled by `4` (and multiplied by `6 * dxyz` to share `L = 24 * dxyz`!):
    //   Actually keep shear scaled by `4` separately so squares are smaller and exact!
    let mut sum_sbar_sq = BigUint::zero();    // sum_cell (sbar_x^2 + sbar_y^2 + sbar_z^2) * L^2
    let mut sum_sbar_tr_sq = BigUint::zero(); // sum_cell (sbar_x + sbar_y + sbar_z)^2 * L^2
    let mut sum_var_sq = BigUint::zero();     // sum_cell [15*(dx^2+dy^2+dz^2) + 4*(qx^2+qy^2+qz^2)] * L^2  (divide by 45!)
    let mut sum_shear_sq = BigUint::zero();   // sum_cell sum_{xy,xz,yz} [9*c0^2 + 3*c1^2 + 3*c2^2 + c3^2] * 16  (divide by 9*16 = 144!)

    let l_i128 = 24 * dxyz;
    let l_val = BigUint::from_u128(l_i128 as u128);
    let l2 = l_val.mul_u(&l_val);
    let fd = BigUint::from_u64(w.flux_den);
    let fd2 = fd.mul_u(&fd);
    let vm_den = l2.mul_u(&fd2).mul_u(&BigUint::from_u64(2)); // 2 * L^2 * flux_den^2

    // Precompute exact integer yield threshold if yield_limit = Some((y_num, y_den)):
    //   cell_vm2_scaled * y_den^2 <= y_num^2 * (2 * L^2 * flux_den^2)
    let yield_check: Option<(BigUint, BigUint)> = s.yield_limit.map(|(y_num, y_den)| {
        let yn = BigUint::from_u64(y_num as u64);
        let yd = BigUint::from_u64(y_den);
        (yd.mul_u(&yd), yn.mul_u(&yn).mul_u(&vm_den))
    });
    let mut max_vm2_scaled = BigUint::zero();

    #[inline]
    fn sq_big(v: i128) -> BigUint {
        let a = v.unsigned_abs();
        let bu = BigUint::from_u128(a);
        bu.mul_u(&bu)
    }

    #[inline]
    fn min3(a: i128, b: i128, c: i128) -> i128 {
        let m = if a < b { a } else { b };
        if m < c { m } else { c }
    }
    #[inline]
    fn max3(a: i128, b: i128, c: i128) -> i128 {
        let m = if a > b { a } else { b };
        if m > c { m } else { c }
    }
    #[inline]
    fn max4_abs(a: i128, b: i128, c: i128, d: i128) -> u128 {
        let m1 = a.unsigned_abs().max(b.unsigned_abs());
        let m2 = c.unsigned_abs().max(d.unsigned_abs());
        m1.max(m2)
    }

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let fx0 = w.fx[idx_fx(ny, nz, i, j, k)] as i128;
                let fx1 = w.fx[idx_fx(ny, nz, i + 1, j, k)] as i128;
                let fy0 = w.fy[idx_fy(ny, nz, i, j, k)] as i128;
                let fy1 = w.fy[idx_fy(ny, nz, i, j + 1, k)] as i128;
                let fz0 = w.fz[idx_fz(ny, nz, i, j, k)] as i128;
                let fz1 = w.fz[idx_fz(ny, nz, i, j, k + 1)] as i128;

                // Shear edge corners:
                // ex (sig_yz): (j,k), (j+1,k), (j,k+1), (j+1,k+1)
                let ex00 = w.ex[idx_ex(ny, nz, i, j, k)] as i128;
                let ex10 = w.ex[idx_ex(ny, nz, i, j + 1, k)] as i128;
                let ex01 = w.ex[idx_ex(ny, nz, i, j, k + 1)] as i128;
                let ex11 = w.ex[idx_ex(ny, nz, i, j + 1, k + 1)] as i128;
                // ey (sig_xz): (i,k), (i+1,k), (i,k+1), (i+1,k+1)
                let ey00 = w.ey[idx_ey(ny, nz, i, j, k)] as i128;
                let ey10 = w.ey[idx_ey(ny, nz, i + 1, j, k)] as i128;
                let ey01 = w.ey[idx_ey(ny, nz, i, j, k + 1)] as i128;
                let ey11 = w.ey[idx_ey(ny, nz, i + 1, j, k + 1)] as i128;
                // ez (sig_xy): (i,j), (i+1,j), (i,j+1), (i+1,j+1)
                let ez00 = w.ez[idx_ez(ny, nz, i, j, k)] as i128;
                let ez10 = w.ez[idx_ez(ny, nz, i + 1, j, k)] as i128;
                let ez01 = w.ez[idx_ez(ny, nz, i, j + 1, k)] as i128;
                let ez11 = w.ez[idx_ez(ny, nz, i + 1, j + 1, k)] as i128;

                // Exact integer cell equilibrium checks (multiplied by 2 * dx * dy * dz):
                let eq_x = 2 * dyz * (fx1 - fx0)
                    + dxz * ((ez01 + ez11) - (ez00 + ez10))
                    + dxy * ((ey01 + ey11) - (ey00 + ey10));
                if eq_x != 0 {
                    return Err(HmzRejection::EquilibriumViolation { i, j, k, axis: 0 });
                }
                let eq_y = 2 * dxz * (fy1 - fy0)
                    + dyz * ((ez10 + ez11) - (ez00 + ez01))
                    + dxy * ((ex01 + ex11) - (ex00 + ex10));
                if eq_y != 0 {
                    return Err(HmzRejection::EquilibriumViolation { i, j, k, axis: 1 });
                }
                let eq_z = 2 * dxy * (fz1 - fz0)
                    + dyz * ((ey10 + ey11) - (ey00 + ey01))
                    + dxz * ((ex10 + ex11) - (ex00 + ex01));
                if eq_z != 0 {
                    return Err(HmzRejection::EquilibriumViolation { i, j, k, axis: 2 });
                }

                // Exact twists for quadratic bubbles q_x, q_y, q_z
                let tw_z = ez00 - ez10 - ez01 + ez11;
                let tw_y = ey00 - ey10 - ey01 + ey11;
                let tw_x = ex00 - ex10 - ex01 + ex11;

                // Scaled by L = 24 * dxyz:
                let qx_l = -3 * dx * (dxz * tw_z + dxy * tw_y);
                let qy_l = -3 * dy * (dyz * tw_z + dxy * tw_x);
                let qz_l = -3 * dz * (dyz * tw_y + dxz * tw_x);

                let sbx_l = 12 * dxyz * (fx0 + fx1) + 2 * dx * (dxz * tw_z + dxy * tw_y);
                let sby_l = 12 * dxyz * (fy0 + fy1) + 2 * dy * (dyz * tw_z + dxy * tw_x);
                let sbz_l = 12 * dxyz * (fz0 + fz1) + 2 * dz * (dyz * tw_y + dxz * tw_x);

                let dx_l = 12 * dxyz * (fx1 - fx0);
                let dy_l = 12 * dxyz * (fy1 - fy0);
                let dz_l = 12 * dxyz * (fz1 - fz0);

                // Exact pointwise 3D von Mises upper bound via 3-point Bernstein normal intervals
                // sigma_aa(xi) in [min(f0*L, m*L - 2*q*L, f1*L), max(...)] + 8-corner convex max!
                let sx_b1 = 12 * dxyz * (fx0 + fx1) - 2 * qx_l;
                let sy_b1 = 12 * dxyz * (fy0 + fy1) - 2 * qy_l;
                let sz_b1 = 12 * dxyz * (fz0 + fz1) - 2 * qz_l;
                let sx_lo = min3(l_i128 * fx0, sx_b1, l_i128 * fx1);
                let sx_hi = max3(l_i128 * fx0, sx_b1, l_i128 * fx1);
                let sy_lo = min3(l_i128 * fy0, sy_b1, l_i128 * fy1);
                let sy_hi = max3(l_i128 * fy0, sy_b1, l_i128 * fy1);
                let sz_lo = min3(l_i128 * fz0, sz_b1, l_i128 * fz1);
                let sz_hi = max3(l_i128 * fz0, sz_b1, l_i128 * fz1);

                let max_ez_abs = max4_abs(ez00, ez10, ez01, ez11);
                let max_ey_abs = max4_abs(ey00, ey10, ey01, ey11);
                let max_ex_abs = max4_abs(ex00, ex10, ex01, ex11);
                let l_u = l_i128 as u128;

                const SAFE_U128: u128 = 1_000_000_000_000_000_000; // 10^18 -> 3*(2e18)^2 + 18*(1e18)^2 = 3e37 < u128::MAX
                let max_n_abs = sx_lo.unsigned_abs().max(sx_hi.unsigned_abs())
                    .max(sy_lo.unsigned_abs().max(sy_hi.unsigned_abs()))
                    .max(sz_lo.unsigned_abs().max(sz_hi.unsigned_abs()));
                let max_sh_u = max_ez_abs.max(max_ey_abs).max(max_ex_abs);
                let fits_u128 = max_n_abs <= SAFE_U128
                    && max_sh_u.checked_mul(l_u).map_or(false, |v| v <= SAFE_U128);

                let cell_vm2_scaled: BigUint = if fits_u128 {
                    let mut dev_max = 0u128;
                    for &sx in &[sx_lo, sx_hi] {
                        for &sy in &[sy_lo, sy_hi] {
                            for &sz in &[sz_lo, sz_hi] {
                                let d1 = (sx - sy).unsigned_abs();
                                let d2 = (sy - sz).unsigned_abs();
                                let d3 = (sz - sx).unsigned_abs();
                                let v = d1 * d1 + d2 * d2 + d3 * d3;
                                if v > dev_max {
                                    dev_max = v;
                                }
                            }
                        }
                    }
                    let ez_l = max_ez_abs * l_u;
                    let ey_l = max_ey_abs * l_u;
                    let ex_l = max_ex_abs * l_u;
                    let sh_term = 6u128 * (ez_l * ez_l + ey_l * ey_l + ex_l * ex_l);
                    BigUint::from_u128(dev_max + sh_term)
                } else {
                    let mut dev_max = BigUint::zero();
                    for &sx in &[sx_lo, sx_hi] {
                        for &sy in &[sy_lo, sy_hi] {
                            for &sz in &[sz_lo, sz_hi] {
                                let v = sq_big(sx - sy).add_u(&sq_big(sy - sz)).add_u(&sq_big(sz - sx));
                                if v > dev_max {
                                    dev_max = v;
                                }
                            }
                        }
                    }
                    let ez_l = BigUint::from_u128(max_ez_abs).mul_u(&l_val);
                    let ey_l = BigUint::from_u128(max_ey_abs).mul_u(&l_val);
                    let ex_l = BigUint::from_u128(max_ex_abs).mul_u(&l_val);
                    let sh_sum = ez_l.mul_u(&ez_l).add_u(&ey_l.mul_u(&ey_l)).add_u(&ex_l.mul_u(&ex_l));
                    dev_max.add_u(&sh_sum.mul_u(&BigUint::from_u64(6)))
                };

                if let Some((ref yd2, ref y_rhs)) = yield_check {
                    if cell_vm2_scaled.mul_u(yd2) > *y_rhs {
                        return Err(HmzRejection::YieldLimitExceeded { i, j, k });
                    }
                }
                if cell_vm2_scaled > max_vm2_scaled {
                    max_vm2_scaled = cell_vm2_scaled;
                }

                sum_sbar_sq = sum_sbar_sq.add_u(&sq_big(sbx_l)).add_u(&sq_big(sby_l)).add_u(&sq_big(sbz_l));
                sum_sbar_tr_sq = sum_sbar_tr_sq.add_u(&sq_big(sbx_l + sby_l + sbz_l));

                let d_sq = sq_big(dx_l).add_u(&sq_big(dy_l)).add_u(&sq_big(dz_l));
                let q_sq = sq_big(qx_l).add_u(&sq_big(qy_l)).add_u(&sq_big(qz_l));
                sum_var_sq = sum_var_sq
                    .add_u(&d_sq.mul_u(&BigUint::from_u64(15)))
                    .add_u(&q_sq.mul_u(&BigUint::from_u64(4)));

                for (e00, e10, e01, e11) in [
                    (ez00, ez10, ez01, ez11),
                    (ey00, ey10, ey01, ey11),
                    (ex00, ex10, ex01, ex11),
                ] {
                    let c0 = e00 + e10 + e01 + e11;
                    let c1 = -e00 + e10 - e01 + e11;
                    let c2 = -e00 - e10 + e01 + e11;
                    let c3 = e00 - e10 - e01 + e11;
                    let sh = sq_big(c0).mul_u(&BigUint::from_u64(9))
                        .add_u(&sq_big(c1).mul_u(&BigUint::from_u64(3)))
                        .add_u(&sq_big(c2).mul_u(&BigUint::from_u64(3)))
                        .add_u(&sq_big(c3));
                    sum_shear_sq = sum_shear_sq.add_u(&sh);
                }
            }
        }
    }

    // 3. Evaluate EXACT rational compliance upper bound at EVERY material uncertainty corner!
    //    Cell volume = dxyz / grid_den^3.
    //    Flux scale squared = flux_den^2.
    //    Normal scale L^2 = (24 * dxyz)^2.
    //    For each corner (mu = mu_num/mat_den, lam = lam_num/mat_den):
    //      a_c = 1/(2*mu) = mat_den / (2 * mu_num)
    //      c_c = lam / (2*mu + 3*lam) = lam_num / (2*mu_num + 3*lam_num)
    //      1 - c_c = (2*mu_num + 2*lam_num) / (2*mu_num + 3*lam_num)
    let vol_cell = BigRat::new(BigInt::from_i128(dxyz), gd2_u.mul_u(&gd_u));

    let r_sbar_sq = BigRat::new(BigInt { neg: false, mag: sum_sbar_sq }, l2.mul_u(&fd2));
    let r_tr_sq = BigRat::new(BigInt { neg: false, mag: sum_sbar_tr_sq }, l2.mul_u(&fd2));
    let r_var_sq = BigRat::new(
        BigInt { neg: false, mag: sum_var_sq },
        l2.mul_u(&fd2).mul_u(&BigUint::from_u64(45)),
    );
    let r_shear_sq = BigRat::new(
        BigInt { neg: false, mag: sum_shear_sq },
        fd2.mul_u(&BigUint::from_u64(144)),
    );
    let max_von_mises_sq = BigRat::new(BigInt { neg: false, mag: max_vm2_scaled }, vm_den);

    let mut compliance_per_corner = Vec::with_capacity(s.material_corners.len());
    let mut worst = BigRat::zero();
    for (ci, mc) in s.material_corners.iter().enumerate() {
        let a_c = BigRat::new(
            BigInt::from_i64(s.mat_den as i64),
            BigUint::from_u64(2 * (mc.mu_num as u64)),
        );
        let denom_bulk = BigUint::from_u128((2i128 * (mc.mu_num as i128) + 3i128 * (mc.lam_num as i128)) as u128);
        let c_c = BigRat::new(BigInt::from_i64(mc.lam_num), denom_bulk.clone());
        let one_minus_c = BigRat::new(
            BigInt::from_i128(2i128 * (mc.mu_num as i128) + 2i128 * (mc.lam_num as i128)),
            denom_bulk,
        );
        let inv_mu = BigRat::new(BigInt::from_i64(s.mat_den as i64), BigUint::from_u64(mc.mu_num as u64));

        // C_corner = vol_cell * [ a_c*(r_sbar_sq - c_c*r_tr_sq) + a_c*(1-c_c)*r_var_sq + inv_mu*r_shear_sq ] + contact_gap_compliance
        let term1 = a_c.mul(&r_sbar_sq.sub(&c_c.mul(&r_tr_sq)));
        let term2 = a_c.mul(&one_minus_c).mul(&r_var_sq);
        let term3 = inv_mu.mul(&r_shear_sq);
        let c_val = vol_cell.mul(&term1.add(&term2).add(&term3)).add(&contact_gap_compliance);
        if ci == 0 || c_val > worst {
            worst = c_val.clone();
        }
        compliance_per_corner.push(c_val);
    }

    Ok(HmzProofReport {
        spec_hash: s.spec_hash(),
        num_cells: nx * ny * nz,
        num_mat_corners: s.material_corners.len(),
        compliance_per_corner,
        worst_compliance_upper: worst,
        max_von_mises_sq,
        contact_gap_compliance,
        contact_faces_verified,
    })
}

/// Compositional Zero-Trust Verifier for two 3D HMZ1 subdomains joined along an x-plane
/// (`A`'s `x_max` side 1 paired with `B`'s `x_min` side 0).
/// Independently verifies BOTH subdomains AND checks exact rational 1-to-1 matching of ALL
/// `(fx, ey, ez)` normal & shear interface tractions!
pub fn verify_composed_hmz_x(
    cert_a: &HmzCert3D,
    cert_b: &HmzCert3D,
) -> Result<HmzComposedReport, HmzRejection> {
    let rep_a = verify_hmz_cert(cert_a)?;
    let rep_b = verify_hmz_cert(cert_b)?;
    let sa = &cert_a.spec;
    let sb = &cert_b.spec;
    if sa.ny != sb.ny || sa.nz != sb.nz || sa.grid_den != sb.grid_den || sa.dy_num != sb.dy_num || sa.dz_num != sb.dz_num {
        return Err(HmzRejection::InterfaceGeometryMismatch);
    }
    let tag_a = match sa.face_kinds[1] {
        BoundaryFaceKind::Interface { tag } => tag,
        _ => return Err(HmzRejection::InterfaceTagMismatch),
    };
    let tag_b = match sb.face_kinds[0] {
        BoundaryFaceKind::Interface { tag } => tag,
        _ => return Err(HmzRejection::InterfaceTagMismatch),
    };
    if tag_a != tag_b {
        return Err(HmzRejection::InterfaceTagMismatch);
    }

    let (ny, nz) = (sa.ny, sa.nz);
    let fa = cert_a.witness.flux_den as i128;
    let fb = cert_b.witness.flux_den as i128;
    let mut matched = 0usize;

    // 1. Normal traction sigma_xx: fx_A(nx_a, j, k) / fa == fx_B(0, j, k) / fb
    for j in 0..ny {
        for k in 0..nz {
            let va = cert_a.witness.fx[idx_fx(ny, nz, sa.nx, j, k)] as i128;
            let vb = cert_b.witness.fx[idx_fx(ny, nz, 0, j, k)] as i128;
            if va * fb != vb * fa {
                return Err(HmzRejection::InterfaceTractionMismatch { kind: 0, idx: j * nz + k });
            }
            matched += 1;
        }
    }
    // 2. Shear traction sigma_xz: ey_A(nx_a, j, k) / fa == ey_B(0, j, k) / fb
    for j in 0..ny {
        for k in 0..=nz {
            let va = cert_a.witness.ey[idx_ey(ny, nz, sa.nx, j, k)] as i128;
            let vb = cert_b.witness.ey[idx_ey(ny, nz, 0, j, k)] as i128;
            if va * fb != vb * fa {
                return Err(HmzRejection::InterfaceTractionMismatch { kind: 1, idx: j * (nz + 1) + k });
            }
            matched += 1;
        }
    }
    // 3. Shear traction sigma_xy: ez_A(nx_a, j, k) / fa == ez_B(0, j, k) / fb
    for j in 0..=ny {
        for k in 0..nz {
            let va = cert_a.witness.ez[idx_ez(ny, nz, sa.nx, j, k)] as i128;
            let vb = cert_b.witness.ez[idx_ez(ny, nz, 0, j, k)] as i128;
            if va * fb != vb * fa {
                return Err(HmzRejection::InterfaceTractionMismatch { kind: 2, idx: j * nz + k });
            }
            matched += 1;
        }
    }

    let total_worst = rep_a.worst_compliance_upper.add(&rep_b.worst_compliance_upper);
    Ok(HmzComposedReport {
        report_a: rep_a,
        report_b: rep_b,
        matched_interface_dofs: matched,
        total_worst_compliance_upper: total_worst,
    })
}
