//! Pure-integer 3D heterogeneous Maxwell magnetostatics (`M5`) and coupled Joule-thermal
//! (`M5 -> M3`) certificate verifier.
//!
//! Operates on the 3D discrete De Rham complex (`ND_0 --curl--> RT_0 --div--> Q_0`):
//! - Dual magnetic field `H in ND_0`: verifies pointwise Ampere conservation `curl(H) == J`
//!   and evaluates `2 W_upper(mu) = int_Omega mu |H|^2 dV` across permeability corners.
//! - Primal vector potential `A in ND_0`: verifies `A x n == 0` on the boundary, computes
//!   `B = curl(A) in RT_0`, and evaluates `2 W_lower(mu) = 2 int J . A - int |B|^2 / mu dV`.
//! - Coupled Joule-thermal (`verify_coupled_em_thermal`): verifies `Q_e_thermal >= (1 / (sigma_e * V_e)) * int_e |J|^2 dV`.
//! - Reluctance torque (`verify_reluctance_torque_bracket`): certifies a two-sided continuum
//!   torque bracket from two rotor positions.

use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};
use crate::sha256::Sha256;
use crate::thermal::{
    idx_qx, idx_qy, idx_qz, verify_thermal_dual_cert, verify_thermal_dual_cert_gdt, GdtScaleBox3D,
    ThermalDualCert3D, ThermalDualReport, ThermalRejection,
};

/// Kernel identity baked into every 3D Maxwell Magnetostatics SHA-256 specification hash.
pub const MAXWELL_KERNEL_ID: &str = "nooterra-kernel/2.3.0 hex3d-nd0-maxwell-exact";

/// One corner of the magnetic permeability uncertainty polytope (`mu_e = mu_num[cell] / mat_den > 0`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxwellMaterialCorner {
    pub mat_den: u64,
    pub mu_num: Vec<i64>, // length nx * ny * nz, strictly positive
}

/// Exact-rational 3D Magnetostatics specification with prescribed `RT_0` coil current density `J`.
/// Current density `J = (jx, jy, jz) / j_den` lives on faces (`RT_0`), satisfying `div(J) = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxwellSpec3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub grid_den: u64,
    pub dx_num: u64,
    pub dy_num: u64,
    pub dz_num: u64,
    pub j_den: u64,
    pub jx: Vec<i64>, // (nx + 1) * ny * nz
    pub jy: Vec<i64>, // nx * (ny + 1) * nz
    pub jz: Vec<i64>, // nx * ny * (nz + 1)
    pub material_corners: Vec<MaxwellMaterialCorner>,
}

impl MaxwellSpec3D {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(MAXWELL_KERNEL_ID.as_bytes());
        h.update_u64(self.nx as u64);
        h.update_u64(self.ny as u64);
        h.update_u64(self.nz as u64);
        h.update_u64(self.grid_den);
        h.update_u64(self.dx_num);
        h.update_u64(self.dy_num);
        h.update_u64(self.dz_num);
        h.update_u64(self.j_den);
        for &v in &self.jx {
            h.update_i64(v);
        }
        for &v in &self.jy {
            h.update_i64(v);
        }
        for &v in &self.jz {
            h.update_i64(v);
        }
        h.update_u64(self.material_corners.len() as u64);
        for mc in &self.material_corners {
            h.update_u64(mc.mat_den);
            h.update_u64(mc.mu_num.len() as u64);
            for &m in &mc.mu_num {
                h.update_i64(m);
            }
        }
        h.finalize()
    }
}

/// Exact-integer 3D Nédélec `ND_0` edge field witness (`V = int_val / field_den`),
/// used for BOTH Dual `H` (Ampère-exact magnetic field) AND Primal `A` (vector potential).
/// Indexing:
///   - `ex[(i * (ny + 1) + j) * (nz + 1) + k]`: `i in 0..nx, j in 0..=ny, k in 0..=nz`
///   - `ey[(i * ny + j) * (nz + 1) + k]`     : `i in 0..=nx, j in 0..ny, k in 0..=nz`
///   - `ez[(i * (ny + 1) + j) * nz + k]`     : `i in 0..=nx, j in 0..=ny, k in 0..nz`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NedelecWitness3D {
    pub field_den: u64,
    pub ex: Vec<i64>,
    pub ey: Vec<i64>,
    pub ez: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxwellDualCert3D {
    pub spec: MaxwellSpec3D,
    pub h_witness: NedelecWitness3D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxwellPrimalCert3D {
    pub spec: MaxwellSpec3D,
    pub a_witness: NedelecWitness3D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaxwellRejection {
    InvalidGrid,
    InvalidDenominator,
    EmptyUncertaintyBox,
    InvalidMaterialParams,
    CurrentSizeMismatch,
    WitnessSizeMismatch,
    AmpereCurlViolation { axis: u8, i: usize, j: usize, k: usize },
    PrimalTangentialBcViolation { axis: u8, i: usize, j: usize, k: usize },
    GeometryMismatch,
    InvalidConductivity,
    CurrentInInsulator { i: usize, j: usize, k: usize },
    JouleSourceUnderestimated { i: usize, j: usize, k: usize },
    ThermalCertRejected(ThermalRejection),
    InvalidTorqueStep,
    CornerCountMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxwellDualReport {
    pub spec_hash: [u8; 32],
    pub num_cells: usize,
    /// Exact rational upper bound `2 * W_mag(mc) = int_Omega mu_mc |H|^2 dV` per permeability corner.
    pub energy2_upper_per_corner: Vec<BigRat>,
    pub worst_energy2_upper: BigRat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxwellPrimalReport {
    pub spec_hash: [u8; 32],
    pub num_cells: usize,
    /// Exact rational lower bound `2 * W_mag(mc) = 2 int J . A - int |curl A|^2 / mu_mc dV` per corner.
    pub energy2_lower_per_corner: Vec<BigRat>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoupledEmThermalReport {
    pub maxwell_report: MaxwellDualReport,
    pub thermal_report: ThermalDualReport,
    /// Exact total electrical I^2 R Joule power `sum_e (1/sigma_e) int_e |J|^2 dV` verified from `M5`.
    pub total_joule_power: BigRat,
    pub active_conductor_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReluctanceTorqueBracket {
    pub w1_lower: BigRat,
    pub w1_upper: BigRat,
    pub w2_lower: BigRat,
    pub w2_upper: BigRat,
    pub torque_lower: BigRat,
    pub torque_upper: BigRat,
}

#[inline]
pub fn idx_edge_x(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * (ny + 1) + j) * (nz + 1) + k
}
#[inline]
pub fn idx_edge_y(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * ny + j) * (nz + 1) + k
}
#[inline]
pub fn idx_edge_z(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * (ny + 1) + j) * nz + k
}

fn validate_maxwell_spec(s: &MaxwellSpec3D) -> Result<(), MaxwellRejection> {
    if s.nx == 0 || s.ny == 0 || s.nz == 0 || s.dx_num == 0 || s.dy_num == 0 || s.dz_num == 0 {
        return Err(MaxwellRejection::InvalidGrid);
    }
    if s.grid_den == 0 || s.j_den == 0 {
        return Err(MaxwellRejection::InvalidDenominator);
    }
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    if s.jx.len() != (nx + 1) * ny * nz
        || s.jy.len() != nx * (ny + 1) * nz
        || s.jz.len() != nx * ny * (nz + 1)
    {
        return Err(MaxwellRejection::CurrentSizeMismatch);
    }
    if s.material_corners.is_empty() {
        return Err(MaxwellRejection::EmptyUncertaintyBox);
    }
    let n_cells = nx * ny * nz;
    for mc in &s.material_corners {
        if mc.mat_den == 0 || mc.mu_num.len() != n_cells {
            return Err(MaxwellRejection::InvalidMaterialParams);
        }
        for &m in &mc.mu_num {
            if m <= 0 {
                return Err(MaxwellRejection::InvalidMaterialParams);
            }
        }
    }
    Ok(())
}

fn validate_nedelec_witness(s: &MaxwellSpec3D, w: &NedelecWitness3D) -> Result<(), MaxwellRejection> {
    if w.field_den == 0 {
        return Err(MaxwellRejection::InvalidDenominator);
    }
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    if w.ex.len() != nx * (ny + 1) * (nz + 1)
        || w.ey.len() != (nx + 1) * ny * (nz + 1)
        || w.ez.len() != (nx + 1) * (ny + 1) * nz
    {
        return Err(MaxwellRejection::WitnessSizeMismatch);
    }
    Ok(())
}

#[inline]
fn sq_big(v: i128) -> BigUint {
    let b = BigUint::from_u128(v.unsigned_abs());
    b.mul_u(&b)
}

#[inline]
fn bil_sq_big(e00: i128, e10: i128, e01: i128, e11: i128) -> BigUint {
    let c0 = e00 + e10 + e01 + e11;
    let c1 = -e00 + e10 - e01 + e11;
    let c2 = -e00 - e10 + e01 + e11;
    let c3 = e00 - e10 - e01 + e11;
    sq_big(c0)
        .mul_u(&BigUint::from_u64(9))
        .add_u(&sq_big(c1).mul_u(&BigUint::from_u64(3)))
        .add_u(&sq_big(c2).mul_u(&BigUint::from_u64(3)))
        .add_u(&sq_big(c3))
}

#[inline]
fn rt0_axis_sq_big(a: &BigInt, b: &BigInt) -> BigUint {
    let t = a.mul(a).add(&a.mul(b)).add(&b.mul(b));
    assert!(!t.neg);
    t.mag
}

/// Pure-integer 3D Nedelec (`ND_0`) dual magnetostatics verifier.
/// Checks pointwise `curl(H) == J` on every face and computes exact rational
/// `2 * W_upper(mu) = int_Omega mu |H|^2 dV` across all permeability corners.
pub fn verify_maxwell_dual_cert(cert: &MaxwellDualCert3D) -> Result<MaxwellDualReport, MaxwellRejection> {
    let (rep, _) = verify_maxwell_dual_axes(cert, None)?;
    Ok(rep)
}

/// 3D Nedelec (`ND_0`) dual verifier over a continuous 3D GD&T tolerance box using covariant
/// curl pullback (`H_i = H_hat_i / alpha_i`) and 8-corner coordinate-wise convexity.
pub fn verify_maxwell_dual_cert_gdt(
    cert: &MaxwellDualCert3D,
    gdt_box: &GdtScaleBox3D,
) -> Result<MaxwellDualReport, MaxwellRejection> {
    if !gdt_box.is_valid() {
        return Err(MaxwellRejection::InvalidDenominator);
    }
    let (rep, _) = verify_maxwell_dual_axes(cert, Some(gdt_box))?;
    Ok(rep)
}

fn eval_nd0_dual_corner(e_axes: &[BigRat; 3], gdt: &GdtScaleBox3D, ax: u64, ay: u64, az: u64) -> BigRat {
    let d = gdt.scale_den as u128;
    let (ax_u, ay_u, az_u) = (ax as u128, ay as u128, az as u128);
    // (ay*az)/(ax*D), (ax*az)/(ay*D), (ax*ay)/(az*D)
    let fx = BigRat::new(BigInt::from_i128((ay_u * az_u) as i128), BigUint::from_u128(ax_u * d));
    let fy = BigRat::new(BigInt::from_i128((ax_u * az_u) as i128), BigUint::from_u128(ay_u * d));
    let fz = BigRat::new(BigInt::from_i128((ax_u * ay_u) as i128), BigUint::from_u128(az_u * d));
    e_axes[0].mul(&fx).add(&e_axes[1].mul(&fy)).add(&e_axes[2].mul(&fz))
}

fn eval_rt0_piola_corner(m_axes: &[BigRat; 3], gdt: &GdtScaleBox3D, ax: u64, ay: u64, az: u64) -> BigRat {
    let d = gdt.scale_den as u128;
    let (ax_u, ay_u, az_u) = (ax as u128, ay as u128, az as u128);
    // (ax*D)/(ay*az), (ay*D)/(ax*az), (az*D)/(ax*ay)
    let fx = BigRat::new(BigInt::from_i128((ax_u * d) as i128), BigUint::from_u128(ay_u * az_u));
    let fy = BigRat::new(BigInt::from_i128((ay_u * d) as i128), BigUint::from_u128(ax_u * az_u));
    let fz = BigRat::new(BigInt::from_i128((az_u * d) as i128), BigUint::from_u128(ax_u * ay_u));
    m_axes[0].mul(&fx).add(&m_axes[1].mul(&fy)).add(&m_axes[2].mul(&fz))
}

fn verify_maxwell_dual_axes(
    cert: &MaxwellDualCert3D,
    gdt_box: Option<&GdtScaleBox3D>,
) -> Result<(MaxwellDualReport, Vec<[BigRat; 3]>), MaxwellRejection> {
    let s = &cert.spec;
    let w = &cert.h_witness;
    validate_maxwell_spec(s)?;
    validate_nedelec_witness(s, w)?;

    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    let dx = BigInt::from_i128(s.dx_num as i128);
    let dy = BigInt::from_i128(s.dy_num as i128);
    let dz = BigInt::from_i128(s.dz_num as i128);
    let dyz = dy.mul(&dz);
    let dxz = dx.mul(&dz);
    let dxy = dx.mul(&dy);
    let dxyz = dxy.mul(&dz);

    let g_jden = BigInt::from_i128((s.grid_den as i128) * (s.j_den as i128));
    let fden = BigInt::from_i128(w.field_den as i128);

    // Pointwise Ampere conservation curl(H) == J scaled to integers:
    //   g * j_den * [dz*(Hz1 - Hz0) - dy*(Hy1 - Hy0)] == dyz * field_den * Jx.
    let rhs_x_scale = dyz.mul(&fden);
    for i in 0..=nx {
        for j in 0..ny {
            for k in 0..nz {
                let dhz = (w.ez[idx_edge_z(ny, nz, i, j + 1, k)] as i128)
                    - (w.ez[idx_edge_z(ny, nz, i, j, k)] as i128);
                let dhy = (w.ey[idx_edge_y(ny, nz, i, j, k + 1)] as i128)
                    - (w.ey[idx_edge_y(ny, nz, i, j, k)] as i128);
                let curl_num = dz.mul(&BigInt::from_i128(dhz)).sub(&dy.mul(&BigInt::from_i128(dhy)));
                let lhs = curl_num.mul(&g_jden);
                let rhs = rhs_x_scale.mul(&BigInt::from_i128(s.jx[idx_qx(ny, nz, i, j, k)] as i128));
                if lhs != rhs {
                    return Err(MaxwellRejection::AmpereCurlViolation { axis: 0, i, j, k });
                }
            }
        }
    }
    // y-face: (Hx(i,j,k+1) - Hx(i,j,k))/dz - (Hz(i+1,j,k) - Hz(i,j,k))/dx == Jy(i,j,k)
    let rhs_y_scale = dxz.mul(&fden);
    for i in 0..nx {
        for j in 0..=ny {
            for k in 0..nz {
                let dhx = (w.ex[idx_edge_x(ny, nz, i, j, k + 1)] as i128)
                    - (w.ex[idx_edge_x(ny, nz, i, j, k)] as i128);
                let dhz = (w.ez[idx_edge_z(ny, nz, i + 1, j, k)] as i128)
                    - (w.ez[idx_edge_z(ny, nz, i, j, k)] as i128);
                let curl_num = dx.mul(&BigInt::from_i128(dhx)).sub(&dz.mul(&BigInt::from_i128(dhz)));
                let lhs = curl_num.mul(&g_jden);
                let rhs = rhs_y_scale.mul(&BigInt::from_i128(s.jy[idx_qy(ny, nz, i, j, k)] as i128));
                if lhs != rhs {
                    return Err(MaxwellRejection::AmpereCurlViolation { axis: 1, i, j, k });
                }
            }
        }
    }
    // z-face: (Hy(i+1,j,k) - Hy(i,j,k))/dx - (Hx(i,j+1,k) - Hx(i,j,k))/dy == Jz(i,j,k)
    let rhs_z_scale = dxy.mul(&fden);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..=nz {
                let dhy = (w.ey[idx_edge_y(ny, nz, i + 1, j, k)] as i128)
                    - (w.ey[idx_edge_y(ny, nz, i, j, k)] as i128);
                let dhx = (w.ex[idx_edge_x(ny, nz, i, j + 1, k)] as i128)
                    - (w.ex[idx_edge_x(ny, nz, i, j, k)] as i128);
                let curl_num = dy.mul(&BigInt::from_i128(dhy)).sub(&dx.mul(&BigInt::from_i128(dhx)));
                let lhs = curl_num.mul(&g_jden);
                let rhs = rhs_z_scale.mul(&BigInt::from_i128(s.jz[idx_qz(ny, nz, i, j, k)] as i128));
                if lhs != rhs {
                    return Err(MaxwellRejection::AmpereCurlViolation { axis: 2, i, j, k });
                }
            }
        }
    }

    // 2. Exact closed-form Nédélec bilinear energy integral per cell separated by axis:
    //    int_cell |H|^2 dV = (vol / (144 * field_den^2)) * [bil_sq(Hx) + bil_sq(Hy) + bil_sq(Hz)]
    let n_cells = nx * ny * nz;
    let mut cell_h2_axes: Vec<[BigUint; 3]> = Vec::with_capacity(n_cells);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let hx_sq = bil_sq_big(
                    w.ex[idx_edge_x(ny, nz, i, j, k)] as i128,
                    w.ex[idx_edge_x(ny, nz, i, j + 1, k)] as i128,
                    w.ex[idx_edge_x(ny, nz, i, j, k + 1)] as i128,
                    w.ex[idx_edge_x(ny, nz, i, j + 1, k + 1)] as i128,
                );
                let hy_sq = bil_sq_big(
                    w.ey[idx_edge_y(ny, nz, i, j, k)] as i128,
                    w.ey[idx_edge_y(ny, nz, i + 1, j, k)] as i128,
                    w.ey[idx_edge_y(ny, nz, i, j, k + 1)] as i128,
                    w.ey[idx_edge_y(ny, nz, i + 1, j, k + 1)] as i128,
                );
                let hz_sq = bil_sq_big(
                    w.ez[idx_edge_z(ny, nz, i, j, k)] as i128,
                    w.ez[idx_edge_z(ny, nz, i + 1, j, k)] as i128,
                    w.ez[idx_edge_z(ny, nz, i, j + 1, k)] as i128,
                    w.ez[idx_edge_z(ny, nz, i + 1, j + 1, k)] as i128,
                );
                cell_h2_axes.push([hx_sq, hy_sq, hz_sq]);
            }
        }
    }

    let gd = BigUint::from_u64(s.grid_den);
    let vol_cell = BigRat::new(dxyz, gd.mul_u(&gd).mul_u(&gd));
    let fd_u = BigUint::from_u64(w.field_den);
    let common_den = fd_u.mul_u(&fd_u).mul_u(&BigUint::from_u64(144));

    let mut energy2_upper_per_corner = Vec::with_capacity(s.material_corners.len());
    let mut axes_per_corner: Vec<[BigRat; 3]> = Vec::with_capacity(s.material_corners.len());
    let mut worst = BigRat::zero();

    for (ci, mc) in s.material_corners.iter().enumerate() {
        let mut weighted: [BigUint; 3] = core::array::from_fn(|_| BigUint::zero());
        for cid in 0..n_cells {
            let mu_u = BigUint::from_u64(mc.mu_num[cid] as u64);
            for ax in 0..3 {
                weighted[ax] = weighted[ax].add_u(&cell_h2_axes[cid][ax].mul_u(&mu_u));
            }
        }
        let den_c = common_den.mul_u(&BigUint::from_u64(mc.mat_den));
        let e_axes: [BigRat; 3] = core::array::from_fn(|ax| {
            vol_cell.mul(&BigRat::new(
                BigInt { neg: false, mag: weighted[ax].clone() },
                den_c.clone(),
            ))
        });
        let e2 = if let Some(gdt) = gdt_box {
            let mut best: Option<BigRat> = None;
            for (ax, ay, az) in gdt.corners() {
                let v = eval_nd0_dual_corner(&e_axes, gdt, ax, ay, az);
                if best.as_ref().map_or(true, |b| &v > b) {
                    best = Some(v);
                }
            }
            best.unwrap()
        } else {
            e_axes[0].add(&e_axes[1]).add(&e_axes[2])
        };
        if ci == 0 || e2 > worst {
            worst = e2.clone();
        }
        energy2_upper_per_corner.push(e2);
        axes_per_corner.push(e_axes);
    }

    Ok((
        MaxwellDualReport {
            spec_hash: s.spec_hash(),
            num_cells: n_cells,
            energy2_upper_per_corner,
            worst_energy2_upper: worst,
        },
        axes_per_corner,
    ))
}

/// Pure-integer 3D Nedelec (`ND_0`) primal magnetostatics verifier.
/// Checks tangential `A x n == 0` on all boundary faces, computes `B = curl(A) in RT_0`, and
/// evaluates `2 * W_lower(mu) = 2 int J . A - int |B|^2 / mu dV <= 2 * W_exact(mu)`.
pub fn verify_maxwell_primal_cert(cert: &MaxwellPrimalCert3D) -> Result<MaxwellPrimalReport, MaxwellRejection> {
    let (rep, _, _) = verify_maxwell_primal_axes(cert, None)?;
    Ok(rep)
}

/// 3D Nedelec (`ND_0`) primal verifier over a continuous 3D GD&T tolerance box.
pub fn verify_maxwell_primal_cert_gdt(
    cert: &MaxwellPrimalCert3D,
    gdt_box: &GdtScaleBox3D,
) -> Result<MaxwellPrimalReport, MaxwellRejection> {
    if !gdt_box.is_valid() {
        return Err(MaxwellRejection::InvalidDenominator);
    }
    let (rep, _, _) = verify_maxwell_primal_axes(cert, Some(gdt_box))?;
    Ok(rep)
}

fn verify_maxwell_primal_axes(
    cert: &MaxwellPrimalCert3D,
    gdt_box: Option<&GdtScaleBox3D>,
) -> Result<(MaxwellPrimalReport, BigRat, Vec<[BigRat; 3]>), MaxwellRejection> {
    let s = &cert.spec;
    let a = &cert.a_witness;
    validate_maxwell_spec(s)?;
    validate_nedelec_witness(s, a)?;

    let (nx, ny, nz) = (s.nx, s.ny, s.nz);

    for i in 0..nx {
        for j in 0..=ny {
            for k in 0..=nz {
                if (j == 0 || j == ny || k == 0 || k == nz) && a.ex[idx_edge_x(ny, nz, i, j, k)] != 0 {
                    return Err(MaxwellRejection::PrimalTangentialBcViolation { axis: 0, i, j, k });
                }
            }
        }
    }
    for i in 0..=nx {
        for j in 0..ny {
            for k in 0..=nz {
                if (i == 0 || i == nx || k == 0 || k == nz) && a.ey[idx_edge_y(ny, nz, i, j, k)] != 0 {
                    return Err(MaxwellRejection::PrimalTangentialBcViolation { axis: 1, i, j, k });
                }
            }
        }
    }
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..nz {
                if (i == 0 || i == nx || j == 0 || j == ny) && a.ez[idx_edge_z(ny, nz, i, j, k)] != 0 {
                    return Err(MaxwellRejection::PrimalTangentialBcViolation { axis: 2, i, j, k });
                }
            }
        }
    }

    // Scaled face curl: B_scaled = B * (dxyz * field_den / grid_den).
    let dx = BigInt::from_i128(s.dx_num as i128);
    let dy = BigInt::from_i128(s.dy_num as i128);
    let dz = BigInt::from_i128(s.dz_num as i128);
    let dxy = dx.mul(&dy);
    let dxz = dx.mul(&dz);
    let dyz = dy.mul(&dz);
    let dxyz = dxy.mul(&dz);

    let bx_at = |i: usize, j: usize, k: usize| -> BigInt {
        let daz = (a.ez[idx_edge_z(ny, nz, i, j + 1, k)] as i128)
            - (a.ez[idx_edge_z(ny, nz, i, j, k)] as i128);
        let day = (a.ey[idx_edge_y(ny, nz, i, j, k + 1)] as i128)
            - (a.ey[idx_edge_y(ny, nz, i, j, k)] as i128);
        dxz.mul(&BigInt::from_i128(daz)).sub(&dxy.mul(&BigInt::from_i128(day)))
    };
    let by_at = |i: usize, j: usize, k: usize| -> BigInt {
        let dax = (a.ex[idx_edge_x(ny, nz, i, j, k + 1)] as i128)
            - (a.ex[idx_edge_x(ny, nz, i, j, k)] as i128);
        let daz = (a.ez[idx_edge_z(ny, nz, i + 1, j, k)] as i128)
            - (a.ez[idx_edge_z(ny, nz, i, j, k)] as i128);
        dxy.mul(&BigInt::from_i128(dax)).sub(&dyz.mul(&BigInt::from_i128(daz)))
    };
    let bz_at = |i: usize, j: usize, k: usize| -> BigInt {
        let day = (a.ey[idx_edge_y(ny, nz, i + 1, j, k)] as i128)
            - (a.ey[idx_edge_y(ny, nz, i, j, k)] as i128);
        let dax = (a.ex[idx_edge_x(ny, nz, i, j + 1, k)] as i128)
            - (a.ex[idx_edge_x(ny, nz, i, j, k)] as i128);
        dyz.mul(&BigInt::from_i128(day)).sub(&dxz.mul(&BigInt::from_i128(dax)))
    };

    let n_cells = nx * ny * nz;
    let mut cell_b2_axes: Vec<[BigUint; 3]> = Vec::with_capacity(n_cells);
    let mut sum_j_dot_a = BigInt::zero();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let bx0 = bx_at(i, j, k); let bx1 = bx_at(i + 1, j, k);
                let by0 = by_at(i, j, k); let by1 = by_at(i, j + 1, k);
                let bz0 = bz_at(i, j, k); let bz1 = bz_at(i, j, k + 1);
                cell_b2_axes.push([
                    rt0_axis_sq_big(&bx0, &bx1),
                    rt0_axis_sq_big(&by0, &by1),
                    rt0_axis_sq_big(&bz0, &bz1),
                ]);

                // Exact separable int_cell J . A dV = (vol / (8 * j_den * field_den)) * sum_axis (J0+J1)*(sum 4 A edges)
                let jx_sum = (s.jx[idx_qx(ny, nz, i, j, k)] as i128)
                    + (s.jx[idx_qx(ny, nz, i + 1, j, k)] as i128);
                let ax_sum = (a.ex[idx_edge_x(ny, nz, i, j, k)] as i128)
                    + (a.ex[idx_edge_x(ny, nz, i, j + 1, k)] as i128)
                    + (a.ex[idx_edge_x(ny, nz, i, j, k + 1)] as i128)
                    + (a.ex[idx_edge_x(ny, nz, i, j + 1, k + 1)] as i128);

                let jy_sum = (s.jy[idx_qy(ny, nz, i, j, k)] as i128)
                    + (s.jy[idx_qy(ny, nz, i, j + 1, k)] as i128);
                let ay_sum = (a.ey[idx_edge_y(ny, nz, i, j, k)] as i128)
                    + (a.ey[idx_edge_y(ny, nz, i + 1, j, k)] as i128)
                    + (a.ey[idx_edge_y(ny, nz, i, j, k + 1)] as i128)
                    + (a.ey[idx_edge_y(ny, nz, i + 1, j, k + 1)] as i128);

                let jz_sum = (s.jz[idx_qz(ny, nz, i, j, k)] as i128)
                    + (s.jz[idx_qz(ny, nz, i, j, k + 1)] as i128);
                let az_sum = (a.ez[idx_edge_z(ny, nz, i, j, k)] as i128)
                    + (a.ez[idx_edge_z(ny, nz, i + 1, j, k)] as i128)
                    + (a.ez[idx_edge_z(ny, nz, i, j + 1, k)] as i128)
                    + (a.ez[idx_edge_z(ny, nz, i + 1, j + 1, k)] as i128);

                let term = BigInt::from_i128(jx_sum).mul(&BigInt::from_i128(ax_sum))
                    .add(&BigInt::from_i128(jy_sum).mul(&BigInt::from_i128(ay_sum)))
                    .add(&BigInt::from_i128(jz_sum).mul(&BigInt::from_i128(az_sum)));
                sum_j_dot_a = sum_j_dot_a.add(&term);
            }
        }
    }

    let gd = BigUint::from_u64(s.grid_den);
    let gd3 = gd.mul_u(&gd).mul_u(&gd);
    let vol_cell = BigRat::new(dxyz.clone(), gd3);
    let fd_u = BigUint::from_u64(a.field_den);
    let fd2 = fd_u.mul_u(&fd_u);

    // 2 * int_Omega J . A dV = (vol_cell / (4 * j_den * field_den)) * sum_j_dot_a
    let work2 = vol_cell.mul(&BigRat::new(
        sum_j_dot_a,
        BigUint::from_u64(4)
            .mul_u(&BigUint::from_u64(s.j_den))
            .mul_u(&fd_u),
    ));

    // int_cell |B|^2 dV = vol_cell * (g^2 / (3 * dxyz^2 * field_den^2)) * cell_b2
    //                   = (1 / (3 * g * dxyz * field_den^2)) * cell_b2
    let b_common_den = BigUint::from_u64(3)
        .mul_u(&gd)
        .mul_u(&dxyz.mag)
        .mul_u(&fd2);

    let mut energy2_lower_per_corner = Vec::with_capacity(s.material_corners.len());
    let mut b_axes_per_corner: Vec<[BigRat; 3]> = Vec::with_capacity(s.material_corners.len());
    for mc in &s.material_corners {
        let mut distinct_mu: Vec<(i64, [BigUint; 3])> = Vec::new();
        for cid in 0..n_cells {
            let mn = mc.mu_num[cid];
            if let Some(slot) = distinct_mu.iter_mut().find(|(m_v, _)| *m_v == mn) {
                for ax in 0..3 {
                    slot.1[ax] = slot.1[ax].add_u(&cell_b2_axes[cid][ax]);
                }
            } else {
                distinct_mu.push((mn, cell_b2_axes[cid].clone()));
            }
        }
        let mat_den_i = BigInt::from_i64(mc.mat_den as i64);
        let mut b_axes: [BigRat; 3] = core::array::from_fn(|_| BigRat::zero());
        for (mn, sum_b2) in distinct_mu {
            let den_m = b_common_den.mul_u(&BigUint::from_u64(mn as u64));
            for ax in 0..3 {
                if !sum_b2[ax].is_zero() {
                    let term = BigRat::new(
                        BigInt { neg: false, mag: sum_b2[ax].clone() }.mul(&mat_den_i),
                        den_m.clone(),
                    );
                    b_axes[ax] = b_axes[ax].add(&term);
                }
            }
        }
        let b_energy = if let Some(gdt) = gdt_box {
            let mut max_b: Option<BigRat> = None;
            for (ax, ay, az) in gdt.corners() {
                let v = eval_rt0_piola_corner(&b_axes, gdt, ax, ay, az);
                if max_b.as_ref().map_or(true, |b| &v > b) {
                    max_b = Some(v);
                }
            }
            max_b.unwrap()
        } else {
            b_axes[0].add(&b_axes[1]).add(&b_axes[2])
        };
        energy2_lower_per_corner.push(work2.sub(&b_energy));
        b_axes_per_corner.push(b_axes);
    }

    Ok((
        MaxwellPrimalReport {
            spec_hash: s.spec_hash(),
            num_cells: n_cells,
            energy2_lower_per_corner,
        },
        work2,
        b_axes_per_corner,
    ))
}

/// Coupled multiphysics verifier (`M5 -> M3`): verifies both the 3D Maxwell dual certificate
/// and the 3D thermal dual certificate on identical geometry, and checks that every cell's
/// thermal source `Q_thermal(e)` bounds the `RT_0` Joule density `(1 / (sigma_e * V_e)) * int_e |J|^2 dV`.
pub fn verify_coupled_em_thermal(
    em_cert: &MaxwellDualCert3D,
    th_cert: &ThermalDualCert3D,
    sigma_num: &[i64], // per-cell electrical conductivity numerator (`0` allowed only where J == 0)
    sigma_den: u64,
) -> Result<CoupledEmThermalReport, MaxwellRejection> {
    if sigma_den == 0 {
        return Err(MaxwellRejection::InvalidConductivity);
    }
    let maxwell_report = verify_maxwell_dual_cert(em_cert)?;
    let thermal_report =
        verify_thermal_dual_cert(th_cert).map_err(MaxwellRejection::ThermalCertRejected)?;

    let es = &em_cert.spec;
    let ts = &th_cert.spec;
    if es.nx != ts.nx
        || es.ny != ts.ny
        || es.nz != ts.nz
        || es.grid_den != ts.grid_den
        || es.dx_num != ts.dx_num
        || es.dy_num != ts.dy_num
        || es.dz_num != ts.dz_num
    {
        return Err(MaxwellRejection::GeometryMismatch);
    }
    let (nx, ny, nz) = (es.nx, es.ny, es.nz);
    let n_cells = nx * ny * nz;
    if sigma_num.len() != n_cells {
        return Err(MaxwellRejection::InvalidConductivity);
    }

    let j_den_u = BigUint::from_u64(es.j_den);
    let j_den2 = j_den_u.mul_u(&j_den_u);
    let three_jden2 = j_den2.mul_u(&BigUint::from_u64(3));
    let sden_i = BigInt::from_i64(sigma_den as i64);

    let gd = BigUint::from_u64(es.grid_den);
    let vol_cell = BigRat::new(
        BigInt::from_i128((es.dx_num as i128) * (es.dy_num as i128) * (es.dz_num as i128)),
        gd.mul_u(&gd).mul_u(&gd),
    );

    let mut total_joule_density_sum = BigRat::zero();
    let mut active_conductor_cells = 0usize;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let cid = (i * ny + j) * nz + k;
                let jx0 = BigInt::from_i64(es.jx[idx_qx(ny, nz, i, j, k)]);
                let jx1 = BigInt::from_i64(es.jx[idx_qx(ny, nz, i + 1, j, k)]);
                let jy0 = BigInt::from_i64(es.jy[idx_qy(ny, nz, i, j, k)]);
                let jy1 = BigInt::from_i64(es.jy[idx_qy(ny, nz, i, j + 1, k)]);
                let jz0 = BigInt::from_i64(es.jz[idx_qz(ny, nz, i, j, k)]);
                let jz1 = BigInt::from_i64(es.jz[idx_qz(ny, nz, i, j, k + 1)]);

                let j2_int = rt0_axis_sq_big(&jx0, &jx1)
                    .add_u(&rt0_axis_sq_big(&jy0, &jy1))
                    .add_u(&rt0_axis_sq_big(&jz0, &jz1));

                let sig_n = sigma_num[cid];
                if sig_n < 0 {
                    return Err(MaxwellRejection::InvalidConductivity);
                }
                if j2_int.is_zero() {
                    continue;
                }
                if sig_n == 0 {
                    return Err(MaxwellRejection::CurrentInInsulator { i, j, k });
                }
                active_conductor_cells += 1;

                // Exact Q_Joule(e) = sigma_den * j2_int / (3 * j_den^2 * sig_n)
                let q_joule = BigRat::new(
                    BigInt { neg: false, mag: j2_int }.mul(&sden_i),
                    three_jden2.mul_u(&BigUint::from_u64(sig_n as u64)),
                );
                let q_therm = BigRat::from_i64(ts.q_src_num[cid], ts.q_src_den);
                if q_therm < q_joule {
                    return Err(MaxwellRejection::JouleSourceUnderestimated { i, j, k });
                }
                total_joule_density_sum = total_joule_density_sum.add(&q_joule);
            }
        }
    }

    let total_joule_power = vol_cell.mul(&total_joule_density_sum);
    Ok(CoupledEmThermalReport {
        maxwell_report,
        thermal_report,
        total_joule_power,
        active_conductor_cells,
    })
}

/// Coupled multiphysics verifier (`M5 -> M3`) over a continuous 3D GD&T tolerance box.
/// Checks both `M5` and `M3` dual certificates over all 8 GD&T corners and verifies that each
/// cell's thermal source dominates worst-case Piola-scaled wire-resistance heating.
pub fn verify_coupled_em_thermal_gdt(
    em_cert: &MaxwellDualCert3D,
    th_cert: &ThermalDualCert3D,
    sigma_num: &[i64],
    sigma_den: u64,
    gdt_box: &GdtScaleBox3D,
) -> Result<CoupledEmThermalReport, MaxwellRejection> {
    if sigma_den == 0 || !gdt_box.is_valid() {
        return Err(MaxwellRejection::InvalidConductivity);
    }
    let maxwell_report = verify_maxwell_dual_cert_gdt(em_cert, gdt_box)?;
    let thermal_report = verify_thermal_dual_cert_gdt(th_cert, gdt_box)
        .map_err(MaxwellRejection::ThermalCertRejected)?;

    let es = &em_cert.spec;
    let ts = &th_cert.spec;
    if es.nx != ts.nx
        || es.ny != ts.ny
        || es.nz != ts.nz
        || es.grid_den != ts.grid_den
        || es.dx_num != ts.dx_num
        || es.dy_num != ts.dy_num
        || es.dz_num != ts.dz_num
    {
        return Err(MaxwellRejection::GeometryMismatch);
    }
    let (nx, ny, nz) = (es.nx, es.ny, es.nz);
    let n_cells = nx * ny * nz;
    if sigma_num.len() != n_cells {
        return Err(MaxwellRejection::InvalidConductivity);
    }

    let j_den_u = BigUint::from_u64(es.j_den);
    let three_jden2 = j_den_u.mul_u(&j_den_u).mul_u(&BigUint::from_u64(3));
    let sden_i = BigInt::from_i64(sigma_den as i64);

    let gd = BigUint::from_u64(es.grid_den);
    let vol_cell = BigRat::new(
        BigInt::from_i128((es.dx_num as i128) * (es.dy_num as i128) * (es.dz_num as i128)),
        gd.mul_u(&gd).mul_u(&gd),
    );

    let mut total_joule_density_sum = BigRat::zero();
    let mut active_conductor_cells = 0usize;
    let corners = gdt_box.corners();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let cid = (i * ny + j) * nz + k;
                let jx0 = BigInt::from_i64(es.jx[idx_qx(ny, nz, i, j, k)]);
                let jx1 = BigInt::from_i64(es.jx[idx_qx(ny, nz, i + 1, j, k)]);
                let jy0 = BigInt::from_i64(es.jy[idx_qy(ny, nz, i, j, k)]);
                let jy1 = BigInt::from_i64(es.jy[idx_qy(ny, nz, i, j + 1, k)]);
                let jz0 = BigInt::from_i64(es.jz[idx_qz(ny, nz, i, j, k)]);
                let jz1 = BigInt::from_i64(es.jz[idx_qz(ny, nz, i, j, k + 1)]);

                let ax_u = rt0_axis_sq_big(&jx0, &jx1);
                let ay_u = rt0_axis_sq_big(&jy0, &jy1);
                let az_u = rt0_axis_sq_big(&jz0, &jz1);

                let sig_n = sigma_num[cid];
                if sig_n < 0 {
                    return Err(MaxwellRejection::InvalidConductivity);
                }
                if ax_u.is_zero() && ay_u.is_zero() && az_u.is_zero() {
                    continue;
                }
                if sig_n == 0 {
                    return Err(MaxwellRejection::CurrentInInsulator { i, j, k });
                }
                active_conductor_cells += 1;

                let cell_den = three_jden2.mul_u(&BigUint::from_u64(sig_n as u64));
                let q_axes = [
                    BigRat::new(BigInt { neg: false, mag: ax_u }.mul(&sden_i), cell_den.clone()),
                    BigRat::new(BigInt { neg: false, mag: ay_u }.mul(&sden_i), cell_den.clone()),
                    BigRat::new(BigInt { neg: false, mag: az_u }.mul(&sden_i), cell_den),
                ];
                let mut q_joule_max = BigRat::zero();
                for &(ax, ay, az) in &corners {
                    let v = eval_rt0_piola_corner(&q_axes, gdt_box, ax, ay, az);
                    if v > q_joule_max {
                        q_joule_max = v;
                    }
                }
                let q_therm = BigRat::from_i64(ts.q_src_num[cid], ts.q_src_den);
                if q_therm < q_joule_max {
                    return Err(MaxwellRejection::JouleSourceUnderestimated { i, j, k });
                }
                total_joule_density_sum = total_joule_density_sum.add(&q_joule_max);
            }
        }
    }

    let total_joule_power = vol_cell.mul(&total_joule_density_sum);
    Ok(CoupledEmThermalReport {
        maxwell_report,
        thermal_report,
        total_joule_power,
        active_conductor_cells,
    })
}

/// Verifies primal (`A`) and dual (`H`) 3D Maxwell certificates at two rotor angles `theta_1`
/// and `theta_2 = theta_1 + dtheta`, returning a two-sided continuum bracket on average
/// reluctance torque `(W(theta_2) - W(theta_1)) / dtheta`.
pub fn verify_reluctance_torque_bracket(
    primal_1: &MaxwellPrimalCert3D,
    dual_1: &MaxwellDualCert3D,
    primal_2: &MaxwellPrimalCert3D,
    dual_2: &MaxwellDualCert3D,
    dtheta_num: i64,
    dtheta_den: u64,
    corner_idx: usize,
) -> Result<ReluctanceTorqueBracket, MaxwellRejection> {
    verify_reluctance_torque_bracket_gdt(
        primal_1,
        dual_1,
        primal_2,
        dual_2,
        dtheta_num,
        dtheta_den,
        corner_idx,
        &GdtScaleBox3D::nominal(),
    )
}

/// Continuum reluctance torque bracket over a continuous 3D GD&T tolerance box using correlated
/// 8-corner coordinate-wise convexity (one manufactured part shares `(alpha_x, alpha_y, alpha_z)`
/// across both rotor angles):
///   `2 * DeltaW_min = work2_2 - max_{c in {-,+}^3} [ M_2(c) + E_1(c) ]`
///   `2 * DeltaW_max = max_{c in {-,+}^3} [ E_2(c) + M_1(c) ] - work2_1`.
#[allow(clippy::too_many_arguments)]
pub fn verify_reluctance_torque_bracket_gdt(
    primal_1: &MaxwellPrimalCert3D,
    dual_1: &MaxwellDualCert3D,
    primal_2: &MaxwellPrimalCert3D,
    dual_2: &MaxwellDualCert3D,
    dtheta_num: i64,
    dtheta_den: u64,
    corner_idx: usize,
    gdt_box: &GdtScaleBox3D,
) -> Result<ReluctanceTorqueBracket, MaxwellRejection> {
    if dtheta_num <= 0 || dtheta_den == 0 || !gdt_box.is_valid() {
        return Err(MaxwellRejection::InvalidTorqueStep);
    }
    if primal_1.spec != dual_1.spec || primal_2.spec != dual_2.spec {
        return Err(MaxwellRejection::GeometryMismatch);
    }
    let (rp1, work2_1, b1_axes) = verify_maxwell_primal_axes(primal_1, Some(gdt_box))?;
    let (rd1, e1_axes) = verify_maxwell_dual_axes(dual_1, Some(gdt_box))?;
    let (rp2, work2_2, b2_axes) = verify_maxwell_primal_axes(primal_2, Some(gdt_box))?;
    let (rd2, e2_axes) = verify_maxwell_dual_axes(dual_2, Some(gdt_box))?;

    if corner_idx >= rp1.energy2_lower_per_corner.len()
        || corner_idx >= rp2.energy2_lower_per_corner.len()
    {
        return Err(MaxwellRejection::CornerCountMismatch);
    }

    let half = BigRat::from_i64(1, 2);
    let w1_lower = rp1.energy2_lower_per_corner[corner_idx].mul(&half);
    let w1_upper = rd1.energy2_upper_per_corner[corner_idx].mul(&half);
    let w2_lower = rp2.energy2_lower_per_corner[corner_idx].mul(&half);
    let w2_upper = rd2.energy2_upper_per_corner[corner_idx].mul(&half);

    // Correlated 8-corner min/max of DeltaW(alpha) over the GD&T box.
    let mut max_m2_plus_e1: Option<BigRat> = None;
    let mut max_e2_plus_m1: Option<BigRat> = None;
    for (ax, ay, az) in gdt_box.corners() {
        let m2 = eval_rt0_piola_corner(&b2_axes[corner_idx], gdt_box, ax, ay, az);
        let e1 = eval_nd0_dual_corner(&e1_axes[corner_idx], gdt_box, ax, ay, az);
        let s_lo = m2.add(&e1);
        if max_m2_plus_e1.as_ref().map_or(true, |b| &s_lo > b) {
            max_m2_plus_e1 = Some(s_lo);
        }

        let e2 = eval_nd0_dual_corner(&e2_axes[corner_idx], gdt_box, ax, ay, az);
        let m1 = eval_rt0_piola_corner(&b1_axes[corner_idx], gdt_box, ax, ay, az);
        let s_hi = e2.add(&m1);
        if max_e2_plus_m1.as_ref().map_or(true, |b| &s_hi > b) {
            max_e2_plus_m1 = Some(s_hi);
        }
    }

    let two_dtheta = BigRat::from_i64(2, 1).mul(&BigRat::from_i64(dtheta_num, dtheta_den));
    let torque_lower = work2_2.sub(&max_m2_plus_e1.unwrap()).div(&two_dtheta);
    let torque_upper = max_e2_plus_m1.unwrap().sub(&work2_1).div(&two_dtheta);

    Ok(ReluctanceTorqueBracket {
        w1_lower,
        w1_upper,
        w2_lower,
        w2_upper,
        torque_lower,
        torque_upper,
    })
}
