//! Zero-Trust 3D Structural Buckling LMI & 3D Circular-Cone Slip Infeasibility Verifier
//! (`#![no_std]`, `#![forbid(unsafe_code)]`, ZERO external dependencies).
//!
//! 1. **`verify_hmz_buckling_lmi`**:
//!    Takes a complete `HmzCert3D` stress certificate plus a claimed buckling safety load factor
//!    `Lambda_cert = lambda_num / lambda_den >= 0`.
//!    - First independently verifies the `HmzCert3D` stress field (`div(sigma*) === 0`, exact BCs,
//!      and exact pointwise 3D von Mises yield `sigma_vM(x) <= sigma_Y`).
//!    - Then INDEPENDENTLY assembles the exact closed-form rational 3D Q1 hexahedral elastic
//!      stiffness `K_3D(mu, lam)` and geometric stiffness `K_G(sigma*)` in pure exact integer
//!      arithmetic (ZERO Gauss quadrature, ZERO floating-point in assembly, ZERO trust in solver!).
//!    - Finally runs the Two-Tier Sound PSD Verifier (Rump formally-verified shifted Cholesky +
//!      exact `BigInt` Bareiss `LDL^T` fallback) on `K_3D(mc) + Lambda_cert * K_G(sigma*) >= 0`
//!      across EVERY material uncertainty corner `mc`!
//!    - Note (Tiered Proof Honesty): `verify_hmz_cert` proves zero-discretization-error continuum
//!      equilibrium & pointwise yield, whereas `verify_hmz_buckling_lmi` proves exact zero-numerical-error
//!      positive definiteness of the discrete 3D Q1 FEM tangent operator (mesh convergence of the
//!      buckling eigenmode is checked separately via refinement).
//!
//! 2. **`verify_hmz_3d_slip_infeasibility`**:
//!    Zero-trust Farkas / Virtual-Work proof that NO statically admissible 3D continuum stress
//!    field exists for a bolted flange spec under 3D circular Coulomb friction (`||tau||_2 <= mu * p_n`).
//!    Catches 3D diagonal slip failures where independent 2D axis checks falsely claim SAFE!

use alloc::vec;
use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};
use crate::cert::BoundaryFaceKind;
use crate::hmz::{
    idx_ex, idx_ey, idx_ez, idx_fx, idx_fy, idx_fz, verify_hmz_cert, HmzCert3D, HmzProofReport,
    HmzRejection, HmzSpec3D,
};
use crate::psd::{verify_psd_i128_scaled, PsdMode, PsdOutcome};
use crate::sha256::Sha256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BucklingRejection {
    StressCertRejected(HmzRejection),
    InvalidLambda,
    NoFreeDofs,
    DofLimitExceeded { n_dof: usize },
    IntegerOverflow,
    BucklingLmiFailed { corner: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BucklingProofReport {
    pub spec_hash: [u8; 32],
    pub lmi_hash: [u8; 32],
    pub stress_report: HmzProofReport,
    pub n_free_dofs: usize,
    pub lambda_cert: BigRat,
    pub corner_outcomes: Vec<PsdOutcome>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlipRejection {
    InvalidSpec,
    MissingContactOrUnconstrainedNeumann,
    InvalidContactParams,
    SlipNotInDualCone,
    NonPositiveVirtualWork,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlipInfeasibilityReport {
    pub spec_hash: [u8; 32],
    pub total_force: [BigRat; 3],
    pub virtual_work: BigRat,
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline]
fn node_signs(a: usize) -> (i128, i128, i128) {
    let sx = if (a >> 2) == 0 { -1 } else { 1 };
    let sy = if ((a >> 1) & 1) == 0 { -1 } else { 1 };
    let sz = if (a & 1) == 0 { -1 } else { 1 };
    (sx, sy, sz)
}

/// Exact integer `24 x 24` element elastic stiffness matrix `K_e_hat = 3456 * g * p^2 * mat_den * K_e`,
/// where `p = dx * dy * dz`. Zero floating-point, zero quadrature error!
pub fn exact_q1_ke_hat(
    dx: i128,
    dy: i128,
    dz: i128,
    mu_num: i128,
    lam_num: i128,
) -> Option<[[i128; 24]; 24]> {
    let p = dx.checked_mul(dy)?.checked_mul(dz)?;
    let dxy = dx.checked_mul(dy)?;
    let dxz = dx.checked_mul(dz)?;
    let dyz = dy.checked_mul(dz)?;
    let w_xx = p.checked_mul(dyz)?.checked_mul(dyz)?;
    let w_yy = p.checked_mul(dxz)?.checked_mul(dxz)?;
    let w_zz = p.checked_mul(dxy)?.checked_mul(dxy)?;
    let w_xy = p.checked_mul(dxy)?.checked_mul(dz)?.checked_mul(dz)?;
    let w_xz = p.checked_mul(dxz)?.checked_mul(dy)?.checked_mul(dy)?;
    let w_yz = p.checked_mul(dyz)?.checked_mul(dx)?.checked_mul(dx)?;

    let mut ke = [[0i128; 24]; 24];
    let mu_plus_lam = mu_num.checked_add(lam_num)?;

    for a in 0..8 {
        let (sax, say, saz) = node_signs(a);
        for b in 0..8 {
            let (sbx, sby, sbz) = node_signs(b);
            let ix = 3 + sax * sbx;
            let iy = 3 + say * sby;
            let iz = 3 + saz * sbz;

            let d_xx = w_xx.checked_mul(sax * sbx * 24 * iy * iz)?;
            let d_yy = w_yy.checked_mul(say * sby * 24 * ix * iz)?;
            let d_zz = w_zz.checked_mul(saz * sbz * 24 * ix * iy)?;

            let d_xy = w_xy.checked_mul(sax * sby * 72 * iz)?;
            let d_yx = w_xy.checked_mul(say * sbx * 72 * iz)?;
            let d_xz = w_xz.checked_mul(sax * sbz * 72 * iy)?;
            let d_zx = w_xz.checked_mul(saz * sbx * 72 * iy)?;
            let d_yz = w_yz.checked_mul(say * sbz * 72 * ix)?;
            let d_zy = w_yz.checked_mul(saz * sby * 72 * ix)?;

            let d_trace = d_xx.checked_add(d_yy)?.checked_add(d_zz)?;
            let base = mu_num.checked_mul(d_trace)?;

            let ra = 3 * a;
            let cb = 3 * b;
            ke[ra][cb] = base.checked_add(mu_plus_lam.checked_mul(d_xx)?)?;
            ke[ra + 1][cb + 1] = base.checked_add(mu_plus_lam.checked_mul(d_yy)?)?;
            ke[ra + 2][cb + 2] = base.checked_add(mu_plus_lam.checked_mul(d_zz)?)?;

            ke[ra][cb + 1] = lam_num.checked_mul(d_xy)?.checked_add(mu_num.checked_mul(d_yx)?)?;
            ke[ra + 1][cb] = lam_num.checked_mul(d_yx)?.checked_add(mu_num.checked_mul(d_xy)?)?;

            ke[ra][cb + 2] = lam_num.checked_mul(d_xz)?.checked_add(mu_num.checked_mul(d_zx)?)?;
            ke[ra + 2][cb] = lam_num.checked_mul(d_zx)?.checked_add(mu_num.checked_mul(d_xz)?)?;

            ke[ra + 1][cb + 2] = lam_num.checked_mul(d_yz)?.checked_add(mu_num.checked_mul(d_zy)?)?;
            ke[ra + 2][cb + 1] = lam_num.checked_mul(d_zy)?.checked_add(mu_num.checked_mul(d_yz)?)?;
        }
    }
    Some(ke)
}

/// Exact integer `8 x 8` scalar element geometric stiffness `G_e_hat = 3456 * g * p^2 * flux_den * G_e`
/// for a cell with HMZ1 stress polynomial `(sbx_l, sby_l, sbz_l, ez, ey, ex)`.
/// The 3D `24 x 24` geometric stiffness is identically `G_e_hat ⊗ I_3`!
#[allow(clippy::too_many_arguments)]
pub fn exact_q1_ge_hat(
    dx: i128,
    dy: i128,
    dz: i128,
    sbx_l: i128,
    sby_l: i128,
    sbz_l: i128,
    ez: [i128; 4],
    ey: [i128; 4],
    ex: [i128; 4],
) -> Option<[[i128; 8]; 8]> {
    let p = dx.checked_mul(dy)?.checked_mul(dz)?;
    let dxy = dx.checked_mul(dy)?;
    let dxz = dx.checked_mul(dz)?;
    let dyz = dy.checked_mul(dz)?;

    let w_n_x = dyz.checked_mul(dyz)?;
    let w_n_y = dxz.checked_mul(dxz)?;
    let w_n_z = dxy.checked_mul(dxy)?;

    let w_s_xy = 2i128.checked_mul(p)?.checked_mul(dxy)?.checked_mul(dz)?.checked_mul(dz)?;
    let w_s_xz = 2i128.checked_mul(p)?.checked_mul(dxz)?.checked_mul(dy)?.checked_mul(dy)?;
    let w_s_yz = 2i128.checked_mul(p)?.checked_mul(dyz)?.checked_mul(dx)?.checked_mul(dx)?;

    let (cz0, cz1, cz2, cz3) = (
        ez[0] + ez[1] + ez[2] + ez[3],
        -ez[0] + ez[1] - ez[2] + ez[3],
        -ez[0] - ez[1] + ez[2] + ez[3],
        ez[0] - ez[1] - ez[2] + ez[3],
    );
    let (cy0, cy1, cy2, cy3) = (
        ey[0] + ey[1] + ey[2] + ey[3],
        -ey[0] + ey[1] - ey[2] + ey[3],
        -ey[0] - ey[1] + ey[2] + ey[3],
        ey[0] - ey[1] - ey[2] + ey[3],
    );
    let (cx0, cx1, cx2, cx3) = (
        ex[0] + ex[1] + ex[2] + ex[3],
        -ex[0] + ex[1] - ex[2] + ex[3],
        -ex[0] - ex[1] + ex[2] + ex[3],
        ex[0] - ex[1] - ex[2] + ex[3],
    );

    let mut ge = [[0i128; 8]; 8];
    for a in 0..8 {
        let (sax, say, saz) = node_signs(a);
        for b in a..8 {
            let (sbx, sby, sbz) = node_signs(b);
            let ix = 3 + sax * sbx;
            let iy = 3 + say * sby;
            let iz = 3 + saz * sbz;

            let mut val = 0i128;
            // Normal terms (sigma_xx, sigma_yy, sigma_zz)
            val = val.checked_add(
                w_n_x.checked_mul(sbx_l)?.checked_mul(sax * sbx * iy * iz)?,
            )?;
            val = val.checked_add(
                w_n_y.checked_mul(sby_l)?.checked_mul(say * sby * ix * iz)?,
            )?;
            val = val.checked_add(
                w_n_z.checked_mul(sbz_l)?.checked_mul(saz * sbz * ix * iy)?,
            )?;

            // Shear xy (ez)
            let sh_xy1 = (sax * sby) * (9 * cz0 + 3 * sbx * cz1 + 3 * say * cz2 + sbx * say * cz3) * iz;
            let sh_xy2 = (say * sbx) * (9 * cz0 + 3 * sax * cz1 + 3 * sby * cz2 + sax * sby * cz3) * iz;
            val = val.checked_add(w_s_xy.checked_mul(sh_xy1.checked_add(sh_xy2)?)?)?;

            // Shear xz (ey)
            let sh_xz1 = (sax * sbz) * (9 * cy0 + 3 * sbx * cy1 + 3 * saz * cy2 + sbx * saz * cy3) * iy;
            let sh_xz2 = (saz * sbx) * (9 * cy0 + 3 * sax * cy1 + 3 * sbz * cy2 + sax * sbz * cy3) * iy;
            val = val.checked_add(w_s_xz.checked_mul(sh_xz1.checked_add(sh_xz2)?)?)?;

            // Shear yz (ex)
            let sh_yz1 = (say * sbz) * (9 * cx0 + 3 * sby * cx1 + 3 * saz * cx2 + sby * saz * cx3) * ix;
            let sh_yz2 = (saz * sby) * (9 * cx0 + 3 * say * cx1 + 3 * sbz * cx2 + say * sbz * cx3) * ix;
            val = val.checked_add(w_s_yz.checked_mul(sh_yz1.checked_add(sh_yz2)?)?)?;

            ge[a][b] = val;
            ge[b][a] = val;
        }
    }
    Some(ge)
}

/// Zero-Trust 3D Structural Buckling LMI Verifier!
/// Independently verifies `cert` (`div(sigma*) === 0`, BCs, exact pointwise 3D von Mises yield),
/// independently assembles exact integer `K_3D(mc) + (lambda_num / lambda_den) * K_G(sigma*)`
/// for EVERY material uncertainty corner `mc`, and certifies positive semidefiniteness via our
/// Two-Tier Sound PSD Verifier (Rump formally-verified shifted Cholesky + exact `BigInt` Bareiss)!
pub fn verify_hmz_buckling_lmi(
    cert: &HmzCert3D,
    lambda_num: i64,
    lambda_den: u64,
    mode: PsdMode,
) -> Result<BucklingProofReport, BucklingRejection> {
    if lambda_num < 0 || lambda_den == 0 {
        return Err(BucklingRejection::InvalidLambda);
    }
    // 1. Independently verify the stress field (equilibrium + BCs + yield)!
    let stress_report =
        verify_hmz_cert(cert).map_err(BucklingRejection::StressCertRejected)?;

    let s = &cert.spec;
    let w = &cert.witness;
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);

    // 2. Map free Q1 nodes (clamped on any Dirichlet face)
    let dir_x0 = matches!(s.face_kinds[0], BoundaryFaceKind::Dirichlet);
    let dir_x1 = matches!(s.face_kinds[1], BoundaryFaceKind::Dirichlet);
    let dir_y0 = matches!(s.face_kinds[2], BoundaryFaceKind::Dirichlet);
    let dir_y1 = matches!(s.face_kinds[3], BoundaryFaceKind::Dirichlet);
    let dir_z0 = matches!(s.face_kinds[4], BoundaryFaceKind::Dirichlet);
    let dir_z1 = matches!(s.face_kinds[5], BoundaryFaceKind::Dirichlet);

    let n_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    let mut node_to_free = vec![usize::MAX; n_nodes];
    let mut n_free = 0usize;
    for ix in 0..=nx {
        for iy in 0..=ny {
            for iz in 0..=nz {
                let fixed = (dir_x0 && ix == 0)
                    || (dir_x1 && ix == nx)
                    || (dir_y0 && iy == 0)
                    || (dir_y1 && iy == ny)
                    || (dir_z0 && iz == 0)
                    || (dir_z1 && iz == nz);
                if !fixed {
                    let nid = (ix * (ny + 1) + iy) * (nz + 1) + iz;
                    node_to_free[nid] = n_free;
                    n_free += 1;
                }
            }
        }
    }
    let n_dof = 3 * n_free;
    if n_dof == 0 {
        return Err(BucklingRejection::NoFreeDofs);
    }
    if n_dof > 4096 {
        return Err(BucklingRejection::DofLimitExceeded { n_dof });
    }

    let dx = s.dx_num as i128;
    let dy = s.dy_num as i128;
    let dz = s.dz_num as i128;
    let dxy = dx * dy;
    let dxz = dx * dz;
    let dyz = dy * dz;
    let dxyz = dxy * dz;

    // 3. Independently assemble exact integer global scalar geometric stiffness G_hat (n_free x n_free)
    let mut g_glob = vec![0i128; n_free * n_free];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let fx0 = w.fx[idx_fx(ny, nz, i, j, k)] as i128;
                let fx1 = w.fx[idx_fx(ny, nz, i + 1, j, k)] as i128;
                let fy0 = w.fy[idx_fy(ny, nz, i, j, k)] as i128;
                let fy1 = w.fy[idx_fy(ny, nz, i, j + 1, k)] as i128;
                let fz0 = w.fz[idx_fz(ny, nz, i, j, k)] as i128;
                let fz1 = w.fz[idx_fz(ny, nz, i, j, k + 1)] as i128;

                let ex = [
                    w.ex[idx_ex(ny, nz, i, j, k)] as i128,
                    w.ex[idx_ex(ny, nz, i, j + 1, k)] as i128,
                    w.ex[idx_ex(ny, nz, i, j, k + 1)] as i128,
                    w.ex[idx_ex(ny, nz, i, j + 1, k + 1)] as i128,
                ];
                let ey = [
                    w.ey[idx_ey(ny, nz, i, j, k)] as i128,
                    w.ey[idx_ey(ny, nz, i + 1, j, k)] as i128,
                    w.ey[idx_ey(ny, nz, i, j, k + 1)] as i128,
                    w.ey[idx_ey(ny, nz, i + 1, j, k + 1)] as i128,
                ];
                let ez = [
                    w.ez[idx_ez(ny, nz, i, j, k)] as i128,
                    w.ez[idx_ez(ny, nz, i + 1, j, k)] as i128,
                    w.ez[idx_ez(ny, nz, i, j + 1, k)] as i128,
                    w.ez[idx_ez(ny, nz, i + 1, j + 1, k)] as i128,
                ];

                let tw_z = ez[0] - ez[1] - ez[2] + ez[3];
                let tw_y = ey[0] - ey[1] - ey[2] + ey[3];
                let tw_x = ex[0] - ex[1] - ex[2] + ex[3];

                let sbx_l = 12 * dxyz * (fx0 + fx1) + 2 * dx * (dxz * tw_z + dxy * tw_y);
                let sby_l = 12 * dxyz * (fy0 + fy1) + 2 * dy * (dyz * tw_z + dxy * tw_x);
                let sbz_l = 12 * dxyz * (fz0 + fz1) + 2 * dz * (dyz * tw_y + dxz * tw_x);

                let ge = exact_q1_ge_hat(dx, dy, dz, sbx_l, sby_l, sbz_l, ez, ey, ex)
                    .ok_or(BucklingRejection::IntegerOverflow)?;

                let mut fnodes = [usize::MAX; 8];
                for a in 0..8 {
                    let ix = i + (a >> 2);
                    let iy = j + ((a >> 1) & 1);
                    let iz = k + (a & 1);
                    fnodes[a] = node_to_free[(ix * (ny + 1) + iy) * (nz + 1) + iz];
                }
                for a in 0..8 {
                    let fa = fnodes[a];
                    if fa == usize::MAX {
                        continue;
                    }
                    for b in 0..8 {
                        let fb = fnodes[b];
                        if fb == usize::MAX {
                            continue;
                        }
                        g_glob[fa * n_free + fb] = g_glob[fa * n_free + fb]
                            .checked_add(ge[a][b])
                            .ok_or(BucklingRejection::IntegerOverflow)?;
                    }
                }
            }
        }
    }

    // Common integer weights for K_3D_hat and K_G_hat (reduced by exact GCD!)
    let wk_raw = (w.flux_den as u128)
        .checked_mul(lambda_den as u128)
        .ok_or(BucklingRejection::IntegerOverflow)?;
    let wg_raw = (s.mat_den as u128)
        .checked_mul(lambda_num as u128)
        .ok_or(BucklingRejection::IntegerOverflow)?;
    let g_wt = gcd_u128(wk_raw, wg_raw).max(1);
    let w_k = i128::try_from(wk_raw / g_wt).map_err(|_| BucklingRejection::IntegerOverflow)?;
    let w_g = i128::try_from(wg_raw / g_wt).map_err(|_| BucklingRejection::IntegerOverflow)?;

    // 4. For EVERY material uncertainty corner, assemble exact M = w_k * K_3D_hat + w_g * (G_hat ⊗ I_3)
    //    and run our Two-Tier Sound PSD Verifier!
    let mut corner_outcomes = Vec::with_capacity(s.material_corners.len());
    let mut mat = vec![0i128; n_dof * n_dof];

    for (ci, mc) in s.material_corners.iter().enumerate() {
        mat.fill(0);
        let ke = exact_q1_ke_hat(dx, dy, dz, mc.mu_num as i128, mc.lam_num as i128)
            .ok_or(BucklingRejection::IntegerOverflow)?;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut fnodes = [usize::MAX; 8];
                    for a in 0..8 {
                        let ix = i + (a >> 2);
                        let iy = j + ((a >> 1) & 1);
                        let iz = k + (a & 1);
                        fnodes[a] = node_to_free[(ix * (ny + 1) + iy) * (nz + 1) + iz];
                    }
                    for a in 0..8 {
                        let fa = fnodes[a];
                        if fa == usize::MAX {
                            continue;
                        }
                        for b in 0..8 {
                            let fb = fnodes[b];
                            if fb == usize::MAX {
                                continue;
                            }
                            for c1 in 0..3 {
                                let row = 3 * fa + c1;
                                for c2 in 0..3 {
                                    let col = 3 * fb + c2;
                                    let term = w_k
                                        .checked_mul(ke[3 * a + c1][3 * b + c2])
                                        .ok_or(BucklingRejection::IntegerOverflow)?;
                                    mat[row * n_dof + col] = mat[row * n_dof + col]
                                        .checked_add(term)
                                        .ok_or(BucklingRejection::IntegerOverflow)?;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Add w_g * (G_glob ⊗ I_3)
        if w_g != 0 {
            for fa in 0..n_free {
                for fb in 0..n_free {
                    let g_val = g_glob[fa * n_free + fb];
                    if g_val == 0 {
                        continue;
                    }
                    let term = w_g.checked_mul(g_val).ok_or(BucklingRejection::IntegerOverflow)?;
                    for c in 0..3 {
                        let row = 3 * fa + c;
                        let col = 3 * fb + c;
                        mat[row * n_dof + col] = mat[row * n_dof + col]
                            .checked_add(term)
                            .ok_or(BucklingRejection::IntegerOverflow)?;
                    }
                }
            }
        }

        let outcome = verify_psd_i128_scaled(&mat, n_dof, mode);
        if !outcome.is_psd() {
            return Err(BucklingRejection::BucklingLmiFailed { corner: ci });
        }
        corner_outcomes.push(outcome);
    }

    let mut lh = Sha256::new();
    lh.update(b"nooterra-kernel/2.2.0 buckling-lmi-q1-hmz1");
    lh.update(&stress_report.spec_hash);
    lh.update_i64(lambda_num);
    lh.update_u64(lambda_den);
    lh.update_u64(n_dof as u64);

    Ok(BucklingProofReport {
        spec_hash: stress_report.spec_hash,
        lmi_hash: lh.finalize(),
        stress_report,
        n_free_dofs: n_dof,
        lambda_cert: BigRat::from_i64(lambda_num, lambda_den),
        corner_outcomes,
    })
}

/// Zero-Trust Exact Farkas / Virtual-Work Proof of 3D Circular-Cone Rigid-Body Slip Infeasibility!
///
/// For a box specification with `Contact { mu = mu_num / mu_den }` on `side 0` (`x = 0`, outward
/// normal `n = (-1, 0, 0)`) and pure `Neumann` tractions on all 5 remaining sides `1..6`:
/// Any statically admissible 3D continuum stress field `sigma in H(div; S)` satisfies:
///   `F_ext · u + int_{x=0} (p_n u_x + tau_y u_y + tau_z u_z) dA = 0`
/// for EVERY rigid-body translation `u = (u_x, u_y, u_z)`.
/// Since `p_n >= 0` and `sqrt(tau_y^2 + tau_z^2) <= mu * p_n`, ANY rigid-body slip witness `u`
/// inside the exact 3D circular dual cone:
///   `u_x >= 0`  AND  `u_x^2 * mu_den^2 >= mu_num^2 * (u_y^2 + u_z^2)`
/// guarantees `(p_n u_x + tau_y u_y + tau_z u_z) >= 0` pointwise on `x = 0`, hence `F_ext · u <= 0`.
/// Therefore, exhibiting an exact integer vector `u` with `F_ext · u > 0` proves with 100%
/// mathematical certainty that NO statically admissible 3D stress field exists (rigid-body 3D slip)!
pub fn verify_hmz_3d_slip_infeasibility(
    spec: &HmzSpec3D,
    slip_u: [i64; 3],
) -> Result<SlipInfeasibilityReport, SlipRejection> {
    if spec.nx == 0
        || spec.ny == 0
        || spec.nz == 0
        || spec.dx_num == 0
        || spec.dy_num == 0
        || spec.dz_num == 0
        || spec.grid_den == 0
    {
        return Err(SlipRejection::InvalidSpec);
    }
    let (mu_num, mu_den) = match spec.face_kinds[0] {
        BoundaryFaceKind::Contact { mu_num, mu_den, gap_den, .. } => {
            if mu_num < 0 || mu_den == 0 || gap_den == 0 {
                return Err(SlipRejection::InvalidContactParams);
            }
            (mu_num, mu_den)
        }
        _ => return Err(SlipRejection::MissingContactOrUnconstrainedNeumann),
    };

    // 1. Verify u lies in the exact 3D circular dual cone: u_x >= 0 and u_x^2 * mu_den^2 >= mu_num^2 * (u_y^2 + u_z^2)
    if slip_u[0] < 0 {
        return Err(SlipRejection::SlipNotInDualCone);
    }
    let ux = BigUint::from_u64(slip_u[0] as u64);
    let uy = BigUint::from_u128((slip_u[1] as i128).unsigned_abs());
    let uz = BigUint::from_u128((slip_u[2] as i128).unsigned_abs());
    let md = BigUint::from_u64(mu_den);
    let mn = BigUint::from_u64(mu_num as u64);
    let lhs = ux.mul_u(&ux).mul_u(&md).mul_u(&md);
    let rhs = mn.mul_u(&mn).mul_u(&uy.mul_u(&uy).add_u(&uz.mul_u(&uz)));
    if lhs < rhs {
        return Err(SlipRejection::SlipNotInDualCone);
    }

    // 2. Independently integrate exact rational total external force F_ext across sides 1..6!
    let lx = BigRat::new(
        BigInt::from_i128((spec.nx as i128) * (spec.dx_num as i128)),
        BigUint::from_u64(spec.grid_den),
    );
    let ly = BigRat::new(
        BigInt::from_i128((spec.ny as i128) * (spec.dy_num as i128)),
        BigUint::from_u64(spec.grid_den),
    );
    let lz = BigRat::new(
        BigInt::from_i128((spec.nz as i128) * (spec.dz_num as i128)),
        BigUint::from_u64(spec.grid_den),
    );
    let area_x = ly.mul(&lz);
    let area_y = lx.mul(&lz);
    let area_z = lx.mul(&ly);
    let side_areas = [&area_x, &area_y, &area_y, &area_z, &area_z]; // sides 1, 2, 3, 4, 5

    let mut total_force = [BigRat::zero(), BigRat::zero(), BigRat::zero()];
    for (idx, area) in side_areas.iter().enumerate() {
        let side = idx + 1;
        match spec.face_kinds[side] {
            BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                if traction_den == 0 {
                    return Err(SlipRejection::InvalidSpec);
                }
                for c in 0..3 {
                    let t = BigRat::from_i64(traction_num[c], traction_den);
                    total_force[c] = total_force[c].add(&area.mul(&t));
                }
            }
            _ => return Err(SlipRejection::MissingContactOrUnconstrainedNeumann),
        }
    }

    // 3. Exact rational virtual work F_ext · u > 0
    let mut work = BigRat::zero();
    for c in 0..3 {
        let uc = BigRat::from_i64(slip_u[c], 1);
        work = work.add(&total_force[c].mul(&uc));
    }
    if work <= BigRat::zero() {
        return Err(SlipRejection::NonPositiveVirtualWork);
    }

    Ok(SlipInfeasibilityReport {
        spec_hash: spec.spec_hash(),
        total_force,
        virtual_work: work,
    })
}
