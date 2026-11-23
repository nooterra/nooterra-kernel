//! Pure-integer 3D heterogeneous conjugate heat transfer (`M3`) certificate verifier.
//!
//! Uses conforming 3D Raviart-Thomas (`RT_0`) heat flux fields `q = (qx, qy, qz) in H(div; Omega)`
//! on structured hexahedral grids `(nx, ny, nz)` to certify continuum upper bounds on thermal
//! compliance, winding average temperature rise, and two-sided hot-spot temperature brackets:
//! - Per-cell heterogeneous thermal conductivity `k_e > 0` across polytopic uncertainty corners.
//! - Per-cell volumetric heat generation `Q_e` (coupled to Joule loss `|J|^2 / sigma`).
//! - Isothermal Dirichlet (`T = T_D`), Neumann (`q . n = q_N`), and Robin convective boundary
//!   conditions (`q . n = h_conv * (T - T_inf)`) with uncertain film coefficient `h_conv`.
//! - Q1 primal temperature lower-bound verifier (`verify_thermal_primal_q1`) for two-sided
//!   continuum hot-spot temperature brackets via the parallelogram law.

use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};
use crate::sha256::Sha256;

/// Kernel identity baked into every 3D Thermal CHT SHA-256 specification hash.
pub const THERMAL_KERNEL_ID: &str = "nooterra-kernel/2.3.0 hex3d-rt0-cht-exact";

/// Boundary condition on each of the 6 box sides (`0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalBcKind {
    /// Isothermal boundary: `T = temp_num / temp_den`.
    Dirichlet { temp_num: i64, temp_den: u64 },
    /// Prescribed outward heat flux: `q . n = flux_num / flux_den` (`0/1` = adiabatic insulated wall).
    Neumann { flux_num: i64, flux_den: u64 },
    /// Convective cooling jacket (Conjugate Heat Transfer): `q . n = h * (T - T_inf)`,
    /// where `h` is specified per uncertainty corner (`corner.robin_h_num[side] / corner.mat_den`).
    Robin { t_inf_num: i64, t_inf_den: u64 },
}

/// One corner of the thermal material & convection uncertainty polytope.
/// Cell `(i, j, k)` uses conductivity `k_num[cell_id] / mat_den > 0`.
/// Robin sides use convective film coefficient `robin_h_num[side] / mat_den > 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalMaterialCorner {
    pub mat_den: u64,
    pub k_num: Vec<i64>,       // length nx * ny * nz, strictly positive
    pub robin_h_num: [i64; 6], // strictly positive on Robin sides
}

/// Exact-rational 3D Heterogeneous Conjugate Heat Transfer specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalSpec3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub grid_den: u64,
    pub dx_num: u64,
    pub dy_num: u64,
    pub dz_num: u64,
    /// Per-cell volumetric heat generation `Q_e = q_src_num[cell] / q_src_den` (W/m^3).
    pub q_src_den: u64,
    pub q_src_num: Vec<i64>,
    pub material_corners: Vec<ThermalMaterialCorner>,
    pub face_bcs: [ThermalBcKind; 6],
}

impl ThermalSpec3D {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(THERMAL_KERNEL_ID.as_bytes());
        h.update_u64(self.nx as u64);
        h.update_u64(self.ny as u64);
        h.update_u64(self.nz as u64);
        h.update_u64(self.grid_den);
        h.update_u64(self.dx_num);
        h.update_u64(self.dy_num);
        h.update_u64(self.dz_num);
        h.update_u64(self.q_src_den);
        h.update_u64(self.q_src_num.len() as u64);
        for &q in &self.q_src_num {
            h.update_i64(q);
        }
        h.update_u64(self.material_corners.len() as u64);
        for mc in &self.material_corners {
            h.update_u64(mc.mat_den);
            h.update_u64(mc.k_num.len() as u64);
            for &k in &mc.k_num {
                h.update_i64(k);
            }
            for &rh in &mc.robin_h_num {
                h.update_i64(rh);
            }
        }
        for bc in &self.face_bcs {
            match *bc {
                ThermalBcKind::Dirichlet { temp_num, temp_den } => {
                    h.update(&[0]);
                    h.update_i64(temp_num);
                    h.update_u64(temp_den);
                }
                ThermalBcKind::Neumann { flux_num, flux_den } => {
                    h.update(&[1]);
                    h.update_i64(flux_num);
                    h.update_u64(flux_den);
                }
                ThermalBcKind::Robin { t_inf_num, t_inf_den } => {
                    h.update(&[2]);
                    h.update_i64(t_inf_num);
                    h.update_u64(t_inf_den);
                }
            }
        }
        h.finalize()
    }
}

/// Exact-integer 3D `RT_0` heat flux witness (`q = int_val / flux_den`).
/// Indexing:
///   - `qx[(i * ny + j) * nz + k]`: `i in 0..=nx, j in 0..ny, k in 0..nz`
///   - `qy[(i * (ny + 1) + j) * nz + k]`: `i in 0..nx, j in 0..=ny, k in 0..nz`
///   - `qz[(i * ny + j) * (nz + 1) + k]`: `i in 0..nx, j in 0..ny, k in 0..=nz`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalFluxWitness3D {
    pub flux_den: u64,
    pub qx: Vec<i64>,
    pub qy: Vec<i64>,
    pub qz: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalDualCert3D {
    pub spec: ThermalSpec3D,
    pub witness: ThermalFluxWitness3D,
}

/// Exact-integer 3D Q1 nodal temperature witness (`T = t_nodes[nid] / temp_den`) for certifying
/// continuum lower bounds `-2 * Pi(T) <= C_th_exact`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalPrimalWitness3D {
    pub temp_den: u64,
    pub t_nodes: Vec<i64>, // length (nx + 1) * (ny + 1) * (nz + 1)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThermalRejection {
    InvalidGrid,
    InvalidDenominator,
    EmptyUncertaintyBox,
    InvalidMaterialParams,
    SourceSizeMismatch,
    WitnessSizeMismatch,
    NeumannFluxMismatch { side: u8, idx: usize },
    DirichletTempMismatch { side: u8, nid: usize },
    DivergenceViolation { i: usize, j: usize, k: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalDualReport {
    pub spec_hash: [u8; 32],
    pub num_cells: usize,
    pub total_heat_power: BigRat,
    /// Exact rational upper bound `C_upper(mc) = int |q|^2/k + int_R (q.n)^2/h + 2 int_D (q.n) T_D + 2 int_R (q.n) T_inf`
    /// per uncertainty corner `mc`.
    pub compliance_upper_per_corner: Vec<BigRat>,
    pub worst_compliance_upper: BigRat,
    /// Exact rational power-weighted average temperature rise upper bound `worst_C / total_heat_power`
    /// (valid when `T_D = T_inf = 0`, `q_N = 0`, and `total_heat_power > 0`).
    pub avg_source_temp_upper: Option<BigRat>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThermalPrimalReport {
    pub spec_hash: [u8; 32],
    pub compliance_lower_per_corner: Vec<BigRat>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HotSpotTempBracket {
    pub region_volume: BigRat,
    pub lower_bound: BigRat,
    pub upper_bound: BigRat,
}

#[inline]
pub fn idx_qx(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * ny + j) * nz + k
}
#[inline]
pub fn idx_qy(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * (ny + 1) + j) * nz + k
}
#[inline]
pub fn idx_qz(ny: usize, nz: usize, i: usize, j: usize, k: usize) -> usize {
    (i * ny + j) * (nz + 1) + k
}
#[inline]
pub fn idx_tnode(ny: usize, nz: usize, ix: usize, iy: usize, iz: usize) -> usize {
    (ix * (ny + 1) + iy) * (nz + 1) + iz
}

fn validate_thermal_spec(s: &ThermalSpec3D) -> Result<(), ThermalRejection> {
    if s.nx == 0 || s.ny == 0 || s.nz == 0 || s.dx_num == 0 || s.dy_num == 0 || s.dz_num == 0 {
        return Err(ThermalRejection::InvalidGrid);
    }
    if s.grid_den == 0 || s.q_src_den == 0 {
        return Err(ThermalRejection::InvalidDenominator);
    }
    let n_cells = s.nx * s.ny * s.nz;
    if s.q_src_num.len() != n_cells {
        return Err(ThermalRejection::SourceSizeMismatch);
    }
    if s.material_corners.is_empty() {
        return Err(ThermalRejection::EmptyUncertaintyBox);
    }
    for bc in &s.face_bcs {
        match *bc {
            ThermalBcKind::Dirichlet { temp_den, .. }
            | ThermalBcKind::Neumann { flux_den: temp_den, .. }
            | ThermalBcKind::Robin { t_inf_den: temp_den, .. } => {
                if temp_den == 0 {
                    return Err(ThermalRejection::InvalidDenominator);
                }
            }
        }
    }
    for mc in &s.material_corners {
        if mc.mat_den == 0 || mc.k_num.len() != n_cells {
            return Err(ThermalRejection::InvalidMaterialParams);
        }
        for &k in &mc.k_num {
            if k <= 0 {
                return Err(ThermalRejection::InvalidMaterialParams);
            }
        }
        for side in 0..6 {
            if matches!(s.face_bcs[side], ThermalBcKind::Robin { .. }) && mc.robin_h_num[side] <= 0 {
                return Err(ThermalRejection::InvalidMaterialParams);
            }
        }
    }
    Ok(())
}

#[inline]
fn sq_u128_big(v: i128) -> BigUint {
    let b = BigUint::from_u128(v.unsigned_abs());
    b.mul_u(&b)
}

#[inline]
fn rt0_axis_energy_int(q0: i128, q1: i128) -> BigUint {
    // q0^2 + q0*q1 + q1^2 = (q0 + q1/2)^2 + 3/4 q1^2 >= 0; computed in BigInt to avoid i128 overflow.
    let b0 = BigInt::from_i128(q0);
    let b1 = BigInt::from_i128(q1);
    let t = b0.mul(&b0).add(&b0.mul(&b1)).add(&b1.mul(&b1));
    assert!(!t.neg);
    t.mag
}

/// Exact rational 3D GD&T dimensional tolerance scale box.
/// Cell dimensions scale as `(alpha_x * dx, alpha_y * dy, alpha_z * dz)` for any
/// `(alpha_x, alpha_y, alpha_z)` in `[ax_min/D, ax_max/D] x [ay_min/D, ay_max/D] x [az_min/D, az_max/D]`.
/// By Piola/Nedelec pullback and coordinate-wise convexity, the worst-case continuum energy over
/// the continuous box is attained at one of the 8 vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdtScaleBox3D {
    pub scale_den: u64,
    pub ax_min: u64,
    pub ax_max: u64,
    pub ay_min: u64,
    pub ay_max: u64,
    pub az_min: u64,
    pub az_max: u64,
}

impl GdtScaleBox3D {
    /// Nominal zero-tolerance box `[1, 1]^3`.
    pub const fn nominal() -> Self {
        Self {
            scale_den: 1,
            ax_min: 1,
            ax_max: 1,
            ay_min: 1,
            ay_max: 1,
            az_min: 1,
            az_max: 1,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.scale_den > 0
            && self.ax_min > 0
            && self.ax_min <= self.ax_max
            && self.ay_min > 0
            && self.ay_min <= self.ay_max
            && self.az_min > 0
            && self.az_min <= self.az_max
    }

    /// Exact rational containment check: `self >= other` (every tolerance point in `other` is covered by `self`).
    pub fn contains(&self, other: &Self) -> bool {
        if !self.is_valid() || !other.is_valid() {
            return false;
        }
        let sd = self.scale_den as u128;
        let od = other.scale_den as u128;
        (self.ax_min as u128) * od <= (other.ax_min as u128) * sd
            && (self.ax_max as u128) * od >= (other.ax_max as u128) * sd
            && (self.ay_min as u128) * od <= (other.ay_min as u128) * sd
            && (self.ay_max as u128) * od >= (other.ay_max as u128) * sd
            && (self.az_min as u128) * od <= (other.az_min as u128) * sd
            && (self.az_max as u128) * od >= (other.az_max as u128) * sd
    }

    /// All 8 vertex triples `(ax, ay, az)` (each divided by `scale_den`).
    pub fn corners(&self) -> [(u64, u64, u64); 8] {
        let xs = [self.ax_min, self.ax_max];
        let ys = [self.ay_min, self.ay_max];
        let zs = [self.az_min, self.az_max];
        let mut out = [(0u64, 0u64, 0u64); 8];
        let mut idx = 0;
        for &x in &xs {
            for &y in &ys {
                for &z in &zs {
                    out[idx] = (x, y, z);
                    idx += 1;
                }
            }
        }
        out
    }

    pub fn hash_into(&self, h: &mut Sha256) {
        h.update_u64(self.scale_den);
        h.update_u64(self.ax_min);
        h.update_u64(self.ax_max);
        h.update_u64(self.ay_min);
        h.update_u64(self.ay_max);
        h.update_u64(self.az_min);
        h.update_u64(self.az_max);
    }
}

/// Pure-integer 3D conjugate heat transfer (`RT_0`) dual verifier.
/// Checks `div(q) == Q_e` in every cell, Neumann heat fluxes, and closed-form rational
/// complementary thermal energy across all uncertainty corners.
pub fn verify_thermal_dual_cert(cert: &ThermalDualCert3D) -> Result<ThermalDualReport, ThermalRejection> {
    verify_thermal_dual_cert_core(cert, None)
}

/// Pure-integer 3D `RT_0` dual thermal verifier over a continuous 3D GD&T tolerance box using
/// contravariant Piola pullback and 8-corner coordinate-wise convexity.
pub fn verify_thermal_dual_cert_gdt(
    cert: &ThermalDualCert3D,
    gdt_box: &GdtScaleBox3D,
) -> Result<ThermalDualReport, ThermalRejection> {
    if !gdt_box.is_valid() {
        return Err(ThermalRejection::InvalidDenominator);
    }
    verify_thermal_dual_cert_core(cert, Some(gdt_box))
}

fn verify_thermal_dual_cert_core(
    cert: &ThermalDualCert3D,
    gdt_box: Option<&GdtScaleBox3D>,
) -> Result<ThermalDualReport, ThermalRejection> {
    let s = &cert.spec;
    let w = &cert.witness;
    validate_thermal_spec(s)?;
    if w.flux_den == 0 {
        return Err(ThermalRejection::InvalidDenominator);
    }
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    if w.qx.len() != (nx + 1) * ny * nz
        || w.qy.len() != nx * (ny + 1) * nz
        || w.qz.len() != nx * ny * (nz + 1)
    {
        return Err(ThermalRejection::WitnessSizeMismatch);
    }

    let mut robin_sq: [BigUint; 6] = core::array::from_fn(|_| BigUint::zero());
    let mut side_sum_qn: [BigInt; 6] = core::array::from_fn(|_| BigInt::zero()); // sum of signed outward q . n (in flux_den units)

    // Side 0: x_min (n = -x), Side 1: x_max (n = +x)
    for (side, i_face, sgn) in [(0usize, 0usize, -1i128), (1usize, nx, 1i128)] {
        for j in 0..ny {
            for k in 0..nz {
                let id = idx_qx(ny, nz, i_face, j, k);
                let qn = sgn * (w.qx[id] as i128);
                match s.face_bcs[side] {
                    ThermalBcKind::Neumann { flux_num, flux_den } => {
                        if qn * (flux_den as i128) != (flux_num as i128) * (w.flux_den as i128) {
                            return Err(ThermalRejection::NeumannFluxMismatch { side: side as u8, idx: id });
                        }
                    }
                    ThermalBcKind::Dirichlet { .. } => {
                        side_sum_qn[side] = side_sum_qn[side].add(&BigInt::from_i128(qn));
                    }
                    ThermalBcKind::Robin { .. } => {
                        side_sum_qn[side] = side_sum_qn[side].add(&BigInt::from_i128(qn));
                        robin_sq[side] = robin_sq[side].add_u(&sq_u128_big(qn));
                    }
                }
            }
        }
    }
    // Side 2: y_min (n = -y), Side 3: y_max (n = +y)
    for (side, j_face, sgn) in [(2usize, 0usize, -1i128), (3usize, ny, 1i128)] {
        for i in 0..nx {
            for k in 0..nz {
                let id = idx_qy(ny, nz, i, j_face, k);
                let qn = sgn * (w.qy[id] as i128);
                match s.face_bcs[side] {
                    ThermalBcKind::Neumann { flux_num, flux_den } => {
                        if qn * (flux_den as i128) != (flux_num as i128) * (w.flux_den as i128) {
                            return Err(ThermalRejection::NeumannFluxMismatch { side: side as u8, idx: id });
                        }
                    }
                    ThermalBcKind::Dirichlet { .. } => {
                        side_sum_qn[side] = side_sum_qn[side].add(&BigInt::from_i128(qn));
                    }
                    ThermalBcKind::Robin { .. } => {
                        side_sum_qn[side] = side_sum_qn[side].add(&BigInt::from_i128(qn));
                        robin_sq[side] = robin_sq[side].add_u(&sq_u128_big(qn));
                    }
                }
            }
        }
    }
    // Side 4: z_min (n = -z), Side 5: z_max (n = +z)
    for (side, k_face, sgn) in [(4usize, 0usize, -1i128), (5usize, nz, 1i128)] {
        for i in 0..nx {
            for j in 0..ny {
                let id = idx_qz(ny, nz, i, j, k_face);
                let qn = sgn * (w.qz[id] as i128);
                match s.face_bcs[side] {
                    ThermalBcKind::Neumann { flux_num, flux_den } => {
                        if qn * (flux_den as i128) != (flux_num as i128) * (w.flux_den as i128) {
                            return Err(ThermalRejection::NeumannFluxMismatch { side: side as u8, idx: id });
                        }
                    }
                    ThermalBcKind::Dirichlet { .. } => {
                        side_sum_qn[side] = side_sum_qn[side].add(&BigInt::from_i128(qn));
                    }
                    ThermalBcKind::Robin { .. } => {
                        side_sum_qn[side] = side_sum_qn[side].add(&BigInt::from_i128(qn));
                        robin_sq[side] = robin_sq[side].add_u(&sq_u128_big(qn));
                    }
                }
            }
        }
    }

    // Pointwise flux conservation div(q) == Q_e scaled to integers:
    //   g * q_src_den * [dyz*(qx1-qx0) + dxz*(qy1-qy0) + dxy*(qz1-qz0)] == dxyz * flux_den * q_src_num.
    let dx = BigInt::from_i128(s.dx_num as i128);
    let dy = BigInt::from_i128(s.dy_num as i128);
    let dz = BigInt::from_i128(s.dz_num as i128);
    let dyz = dy.mul(&dz);
    let dxz = dx.mul(&dz);
    let dxy = dx.mul(&dy);
    let dxyz = dxy.mul(&dz);

    let lhs_scale = BigInt::from_i128((s.grid_den as i128) * (s.q_src_den as i128));
    let rhs_scale = dxyz.mul(&BigInt::from_i128(w.flux_den as i128));

    let n_cells = nx * ny * nz;
    let mut cell_e_axes: Vec<[BigUint; 3]> = Vec::with_capacity(n_cells);
    let mut sum_q_src = BigInt::zero();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let cid = (i * ny + j) * nz + k;
                let qx0 = w.qx[idx_qx(ny, nz, i, j, k)] as i128;
                let qx1 = w.qx[idx_qx(ny, nz, i + 1, j, k)] as i128;
                let qy0 = w.qy[idx_qy(ny, nz, i, j, k)] as i128;
                let qy1 = w.qy[idx_qy(ny, nz, i, j + 1, k)] as i128;
                let qz0 = w.qz[idx_qz(ny, nz, i, j, k)] as i128;
                let qz1 = w.qz[idx_qz(ny, nz, i, j, k + 1)] as i128;

                let div_int = dyz
                    .mul(&BigInt::from_i128(qx1 - qx0))
                    .add(&dxz.mul(&BigInt::from_i128(qy1 - qy0)))
                    .add(&dxy.mul(&BigInt::from_i128(qz1 - qz0)));
                let lhs = div_int.mul(&lhs_scale);
                let rhs = rhs_scale.mul(&BigInt::from_i128(s.q_src_num[cid] as i128));
                if lhs != rhs {
                    return Err(ThermalRejection::DivergenceViolation { i, j, k });
                }

                sum_q_src = sum_q_src.add(&BigInt::from_i128(s.q_src_num[cid] as i128));

                cell_e_axes.push([
                    rt0_axis_energy_int(qx0, qx1),
                    rt0_axis_energy_int(qy0, qy1),
                    rt0_axis_energy_int(qz0, qz1),
                ]);
            }
        }
    }

    let gd = BigUint::from_u64(s.grid_den);
    let gd2 = gd.mul_u(&gd);
    let gd3 = gd2.mul_u(&gd);
    let fd = BigUint::from_u64(w.flux_den);
    let fd2 = fd.mul_u(&fd);

    let vol_cell = BigRat::new(dxyz.clone(), gd3);
    let total_heat_power = vol_cell.mul(&BigRat::new(sum_q_src, BigUint::from_u64(s.q_src_den)));

    let area_sides = [
        BigRat::new(dyz.clone(), gd2.clone()),
        BigRat::new(dyz, gd2.clone()),
        BigRat::new(dxz.clone(), gd2.clone()),
        BigRat::new(dxz, gd2.clone()),
        BigRat::new(dxy.clone(), gd2.clone()),
        BigRat::new(dxy, gd2),
    ];

    // Linear boundary work term (independent of k, h): 2 * sum_D (q.n)*T_D + 2 * sum_R (q.n)*T_inf
    let mut bnd_linear = BigRat::zero();
    for side in 0..6 {
        let t_bnd = match s.face_bcs[side] {
            ThermalBcKind::Dirichlet { temp_num, temp_den } => Some((temp_num, temp_den)),
            ThermalBcKind::Robin { t_inf_num, t_inf_den } => Some((t_inf_num, t_inf_den)),
            ThermalBcKind::Neumann { .. } => None,
        };
        if let Some((tn, td)) = t_bnd {
            if tn != 0 && !side_sum_qn[side].is_zero() {
                let qn_int = BigRat::new(side_sum_qn[side].clone(), fd.clone());
                let t_rat = BigRat::from_i64(tn, td);
                let term = BigRat::from_i64(2, 1)
                    .mul(&area_sides[side])
                    .mul(&qn_int)
                    .mul(&t_rat);
                bnd_linear = bnd_linear.add(&term);
            }
        }
    }

    let cell_scale_den = fd2.mul_u(&BigUint::from_u64(3));
    let mut compliance_upper_per_corner = Vec::with_capacity(s.material_corners.len());
    let mut worst = BigRat::zero();

    for (ci, mc) in s.material_corners.iter().enumerate() {
        let mut distinct_k: Vec<(i64, [BigUint; 3])> = Vec::new();
        for cid in 0..n_cells {
            let kn = mc.k_num[cid];
            if let Some(slot) = distinct_k.iter_mut().find(|(k_v, _)| *k_v == kn) {
                slot.1[0] = slot.1[0].add_u(&cell_e_axes[cid][0]);
                slot.1[1] = slot.1[1].add_u(&cell_e_axes[cid][1]);
                slot.1[2] = slot.1[2].add_u(&cell_e_axes[cid][2]);
            } else {
                distinct_k.push((kn, cell_e_axes[cid].clone()));
            }
        }
        let mat_den_i = BigInt::from_i64(mc.mat_den as i64);
        let mut vol_axis: [BigRat; 3] = core::array::from_fn(|_| BigRat::zero());
        for (kn, sum_e) in distinct_k {
            let k_den = cell_scale_den.mul_u(&BigUint::from_u64(kn as u64));
            for ax in 0..3 {
                if !sum_e[ax].is_zero() {
                    let term = BigRat::new(
                        BigInt { neg: false, mag: sum_e[ax].clone() }.mul(&mat_den_i),
                        k_den.clone(),
                    );
                    vol_axis[ax] = vol_axis[ax].add(&term);
                }
            }
        }
        for ax in 0..3 {
            vol_axis[ax] = vol_axis[ax].mul(&vol_cell);
        }

        // Robin convective boundary energy separated by normal axis (0:x, 1:y, 2:z)
        let mut rob_axis: [BigRat; 3] = core::array::from_fn(|_| BigRat::zero());
        for side in 0..6 {
            if matches!(s.face_bcs[side], ThermalBcKind::Robin { .. }) && !robin_sq[side].is_zero() {
                let term = BigRat::new(
                    BigInt { neg: false, mag: robin_sq[side].clone() }.mul(&mat_den_i),
                    fd2.mul_u(&BigUint::from_u64(mc.robin_h_num[side] as u64)),
                );
                let ax = side / 2;
                rob_axis[ax] = rob_axis[ax].add(&area_sides[side].mul(&term));
            }
        }

        let c_val = if let Some(gdt) = gdt_box {
            let d = gdt.scale_den as u128;
            let d2 = BigInt::from_i128((d * d) as i128);
            let mut best_corner: Option<BigRat> = None;
            for (ax, ay, az) in gdt.corners() {
                let (ax_u, ay_u, az_u) = (ax as u128, ay as u128, az as u128);
                // Volume Piola factors: ax/(ay*az) = (ax*D)/(ay*az), etc.
                let fx = BigRat::new(BigInt::from_i128((ax_u * d) as i128), BigUint::from_u128(ay_u * az_u));
                let fy = BigRat::new(BigInt::from_i128((ay_u * d) as i128), BigUint::from_u128(ax_u * az_u));
                let fz = BigRat::new(BigInt::from_i128((az_u * d) as i128), BigUint::from_u128(ax_u * ay_u));
                // Robin Piola factors: 1/(ay*az) = D^2/(ay*az), etc.
                let rx = BigRat::new(d2.clone(), BigUint::from_u128(ay_u * az_u));
                let ry = BigRat::new(d2.clone(), BigUint::from_u128(ax_u * az_u));
                let rz = BigRat::new(d2.clone(), BigUint::from_u128(ax_u * ay_u));
                let val = vol_axis[0].mul(&fx)
                    .add(&vol_axis[1].mul(&fy))
                    .add(&vol_axis[2].mul(&fz))
                    .add(&rob_axis[0].mul(&rx))
                    .add(&rob_axis[1].mul(&ry))
                    .add(&rob_axis[2].mul(&rz))
                    .add(&bnd_linear);
                if best_corner.as_ref().map_or(true, |b| &val > b) {
                    best_corner = Some(val);
                }
            }
            best_corner.unwrap()
        } else {
            vol_axis[0]
                .add(&vol_axis[1])
                .add(&vol_axis[2])
                .add(&rob_axis[0])
                .add(&rob_axis[1])
                .add(&rob_axis[2])
                .add(&bnd_linear)
        };

        if ci == 0 || c_val > worst {
            worst = c_val.clone();
        }
        compliance_upper_per_corner.push(c_val);
    }

    let homogeneous_bcs = s.face_bcs.iter().all(|bc| match *bc {
        ThermalBcKind::Dirichlet { temp_num, .. } => temp_num == 0,
        ThermalBcKind::Neumann { flux_num, .. } => flux_num == 0,
        ThermalBcKind::Robin { t_inf_num, .. } => t_inf_num == 0,
    });
    let avg_source_temp_upper = if homogeneous_bcs && total_heat_power > BigRat::zero() {
        Some(worst.div(&total_heat_power))
    } else {
        None
    };

    Ok(ThermalDualReport {
        spec_hash: s.spec_hash(),
        num_cells: n_cells,
        total_heat_power,
        compliance_upper_per_corner,
        worst_compliance_upper: worst,
        avg_source_temp_upper,
    })
}

/// Pure-integer 3D Q1 primal temperature verifier.
/// Checks Dirichlet BCs and evaluates the exact rational continuum lower bound
/// `-2 * Pi(T) = 2 * int Q T - 2 * int_N q_N T - int k |grad T|^2 - int_R h (T - T_inf)^2 <= C_exact`
/// via closed-form Q1 trilinear polynomial integrals.
pub fn verify_thermal_primal_q1(
    spec: &ThermalSpec3D,
    witness: &ThermalPrimalWitness3D,
) -> Result<ThermalPrimalReport, ThermalRejection> {
    validate_thermal_spec(spec)?;
    if witness.temp_den == 0 {
        return Err(ThermalRejection::InvalidDenominator);
    }
    let (nx, ny, nz) = (spec.nx, spec.ny, spec.nz);
    let n_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    if witness.t_nodes.len() != n_nodes {
        return Err(ThermalRejection::WitnessSizeMismatch);
    }

    // 1. Check exact Dirichlet nodal values on any Dirichlet side
    let check_dir = |side: u8, ix: usize, iy: usize, iz: usize, tn: i64, td: u64| -> Result<(), ThermalRejection> {
        let nid = idx_tnode(ny, nz, ix, iy, iz);
        let lhs = (witness.t_nodes[nid] as i128) * (td as i128);
        let rhs = (tn as i128) * (witness.temp_den as i128);
        if lhs != rhs {
            Err(ThermalRejection::DirichletTempMismatch { side, nid })
        } else {
            Ok(())
        }
    };
    for side in 0..6 {
        if let ThermalBcKind::Dirichlet { temp_num, temp_den } = spec.face_bcs[side] {
            match side {
                0 | 1 => {
                    let ix = if side == 0 { 0 } else { nx };
                    for iy in 0..=ny {
                        for iz in 0..=nz {
                            check_dir(side as u8, ix, iy, iz, temp_num, temp_den)?;
                        }
                    }
                }
                2 | 3 => {
                    let iy = if side == 2 { 0 } else { ny };
                    for ix in 0..=nx {
                        for iz in 0..=nz {
                            check_dir(side as u8, ix, iy, iz, temp_num, temp_den)?;
                        }
                    }
                }
                _ => {
                    let iz = if side == 4 { 0 } else { nz };
                    for ix in 0..=nx {
                        for iy in 0..=ny {
                            check_dir(side as u8, ix, iy, iz, temp_num, temp_den)?;
                        }
                    }
                }
            }
        }
    }

    // Closed-form Q1 gradient integral via the four edge differences along each axis:
    //   int_cell (dT/dx)^2 dV = (dy*dz / (144 * dx)) * (9 c0^2 + 3 c1^2 + 3 c2^2 + c3^2).
    let dx = spec.dx_num as i128;
    let dy = spec.dy_num as i128;
    let dz = spec.dz_num as i128;
    let dyz = dy * dz;
    let dxz = dx * dz;
    let dxy = dx * dy;
    let dxyz = dxy * dz;

    let bil_sq = |e00: i128, e10: i128, e01: i128, e11: i128| -> BigUint {
        let c0 = e00 + e10 + e01 + e11;
        let c1 = -e00 + e10 - e01 + e11;
        let c2 = -e00 - e10 + e01 + e11;
        let c3 = e00 - e10 - e01 + e11;
        sq_u128_big(c0)
            .mul_u(&BigUint::from_u64(9))
            .add_u(&sq_u128_big(c1).mul_u(&BigUint::from_u64(3)))
            .add_u(&sq_u128_big(c2).mul_u(&BigUint::from_u64(3)))
            .add_u(&sq_u128_big(c3))
    };

    let n_cells = nx * ny * nz;
    let mut cell_grad_hat: Vec<BigUint> = Vec::with_capacity(n_cells); // scaled by 144 * dxyz * g * temp_den^2!
    let mut sum_q_t = BigInt::zero(); // sum_cell q_src_num[c] * (sum 8 corners t)

    let w_gx = BigUint::from_u128((dyz * dyz) as u128);
    let w_gy = BigUint::from_u128((dxz * dxz) as u128);
    let w_gz = BigUint::from_u128((dxy * dxy) as u128);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let cid = (i * ny + j) * nz + k;
                let t = |di: usize, dj: usize, dk: usize| -> i128 {
                    witness.t_nodes[idx_tnode(ny, nz, i + di, j + dj, k + dk)] as i128
                };
                let t000 = t(0, 0, 0); let t001 = t(0, 0, 1);
                let t010 = t(0, 1, 0); let t011 = t(0, 1, 1);
                let t100 = t(1, 0, 0); let t101 = t(1, 0, 1);
                let t110 = t(1, 1, 0); let t111 = t(1, 1, 1);

                let gx = bil_sq(t100 - t000, t110 - t010, t101 - t001, t111 - t011);
                let gy = bil_sq(t010 - t000, t110 - t100, t011 - t001, t111 - t101);
                let gz = bil_sq(t001 - t000, t101 - t100, t011 - t010, t111 - t110);
                let g_tot = gx.mul_u(&w_gx).add_u(&gy.mul_u(&w_gy)).add_u(&gz.mul_u(&w_gz));
                cell_grad_hat.push(g_tot);

                let t_sum = t000 + t001 + t010 + t011 + t100 + t101 + t110 + t111;
                sum_q_t = sum_q_t.add(&BigInt::from_i128(spec.q_src_num[cid] as i128).mul(&BigInt::from_i128(t_sum)));
            }
        }
    }

    let gd = BigUint::from_u64(spec.grid_den);
    let gd2 = gd.mul_u(&gd);
    let gd3 = gd2.mul_u(&gd);
    let td = BigUint::from_u64(witness.temp_den);
    let td2 = td.mul_u(&td);

    // 2 * int_Omega Q * T dV = 2 * (dxyz / g^3) * (1 / 8) * sum_q_t / (q_src_den * temp_den)
    let source_work_2 = BigRat::new(
        BigInt::from_i128(dxyz).mul(&sum_q_t),
        gd3.mul_u(&BigUint::from_u64(4))
            .mul_u(&BigUint::from_u64(spec.q_src_den))
            .mul_u(&td),
    );

    // Boundary integrals: Neumann (-2 int_N q_N T) and Robin (-int_R h (T - T_inf)^2)
    let mut neumann_work_2 = BigRat::zero();
    let mut robin_face_int: [BigRat; 6] = core::array::from_fn(|_| BigRat::zero()); // exact int_{side} (T - T_inf)^2 dA

    for side in 0..6 {
        let area_cell = match side {
            0 | 1 => BigRat::new(BigInt::from_i128(dyz), gd2.clone()),
            2 | 3 => BigRat::new(BigInt::from_i128(dxz), gd2.clone()),
            _ => BigRat::new(BigInt::from_i128(dxy), gd2.clone()),
        };
        let (nu, nv) = match side {
            0 | 1 => (ny, nz),
            2 | 3 => (nx, nz),
            _ => (nx, ny),
        };
        let get_t = |u: usize, v: usize| -> i128 {
            let (ix, iy, iz) = match side {
                0 => (0, u, v),
                1 => (nx, u, v),
                2 => (u, 0, v),
                3 => (u, ny, v),
                4 => (u, v, 0),
                _ => (u, v, nz),
            };
            witness.t_nodes[idx_tnode(ny, nz, ix, iy, iz)] as i128
        };
        match spec.face_bcs[side] {
            ThermalBcKind::Neumann { flux_num, flux_den } if flux_num != 0 => {
                let mut sum_t4 = BigInt::zero();
                for u in 0..nu {
                    for v in 0..nv {
                        let s4 = get_t(u, v) + get_t(u + 1, v) + get_t(u, v + 1) + get_t(u + 1, v + 1);
                        sum_t4 = sum_t4.add(&BigInt::from_i128(s4));
                    }
                }
                // 2 * int q_N T = 2 * area_cell * (flux_num/flux_den) * (sum_t4 / (4 * temp_den))
                let term = area_cell
                    .mul(&BigRat::from_i64(flux_num, flux_den))
                    .mul(&BigRat::new(sum_t4, td.mul_u(&BigUint::from_u64(2))));
                neumann_work_2 = neumann_work_2.add(&term);
            }
            ThermalBcKind::Robin { t_inf_num, t_inf_den } => {
                // On each 2D quad, (T - T_inf) = (t_int * t_inf_den - t_inf_num * temp_den) / (temp_den * t_inf_den)
                let mut sum_sq = BigUint::zero();
                let tid = t_inf_den as i128;
                let tin_td = (t_inf_num as i128) * (witness.temp_den as i128);
                for u in 0..nu {
                    for v in 0..nv {
                        let e00 = get_t(u, v) * tid - tin_td;
                        let e10 = get_t(u + 1, v) * tid - tin_td;
                        let e01 = get_t(u, v + 1) * tid - tin_td;
                        let e11 = get_t(u + 1, v + 1) * tid - tin_td;
                        sum_sq = sum_sq.add_u(&bil_sq(e00, e10, e01, e11));
                    }
                }
                let den = td2
                    .mul_u(&BigUint::from_u64(t_inf_den))
                    .mul_u(&BigUint::from_u64(t_inf_den))
                    .mul_u(&BigUint::from_u64(144));
                robin_face_int[side] = area_cell.mul(&BigRat::new(BigInt { neg: false, mag: sum_sq }, den));
            }
            _ => {}
        }
    }

    let grad_common_den = BigUint::from_u128(144u128 * (dxyz as u128))
        .mul_u(&gd)
        .mul_u(&td2);
    let mut compliance_lower_per_corner = Vec::with_capacity(spec.material_corners.len());

    for mc in &spec.material_corners {
        let mut grad_sum = BigUint::zero();
        for cid in 0..n_cells {
            let kn = BigUint::from_u64(mc.k_num[cid] as u64);
            grad_sum = grad_sum.add_u(&cell_grad_hat[cid].mul_u(&kn));
        }
        let grad_energy = BigRat::new(
            BigInt { neg: false, mag: grad_sum },
            grad_common_den.mul_u(&BigUint::from_u64(mc.mat_den)),
        );
        let mut rob_energy = BigRat::zero();
        for side in 0..6 {
            if matches!(spec.face_bcs[side], ThermalBcKind::Robin { .. }) {
                let h_rat = BigRat::from_i64(mc.robin_h_num[side], mc.mat_den);
                rob_energy = rob_energy.add(&h_rat.mul(&robin_face_int[side]));
            }
        }
        let c_lo = source_work_2.sub(&neumann_work_2).sub(&grad_energy).sub(&rob_energy);
        compliance_lower_per_corner.push(c_lo);
    }

    Ok(ThermalPrimalReport {
        spec_hash: spec.spec_hash(),
        compliance_lower_per_corner,
    })
}

/// Two-sided continuum hot-spot temperature bracket via the parallelogram law.
///
/// For homogeneous BCs (`T_D = T_inf = 0`, `q_N = 0`) and target cell `(hi, hj, hk)`, let
/// `Q_adj = (1 / V_cell) * 1_{hot}` and `alpha = alpha_num / alpha_den > 0`. By bilinear
/// symmetry of the 3D CHT PDE:
///   `T_hot_avg = (1 / (4 * alpha)) * [ C(Q + alpha * Q_adj) - C(Q - alpha * Q_adj) ]`.
/// Combines `dual_plus` with `primal_minus` for the upper bound and `primal_plus` with
/// `dual_minus` for the lower bound at `corner_idx`.
#[allow(clippy::too_many_arguments)]
pub fn verify_hotspot_temp_bracket(
    base_spec: &ThermalSpec3D,
    hot_cell: (usize, usize, usize),
    alpha_num: i64,
    alpha_den: u64,
    dual_plus: &ThermalDualCert3D,
    primal_plus: &ThermalPrimalWitness3D,
    dual_minus: &ThermalDualCert3D,
    primal_minus: &ThermalPrimalWitness3D,
    corner_idx: usize,
) -> Result<HotSpotTempBracket, ThermalRejection> {
    validate_thermal_spec(base_spec)?;
    if alpha_num <= 0 || alpha_den == 0 || corner_idx >= base_spec.material_corners.len() {
        return Err(ThermalRejection::InvalidDenominator);
    }
    let (hi, hj, hk) = hot_cell;
    if hi >= base_spec.nx || hj >= base_spec.ny || hk >= base_spec.nz {
        return Err(ThermalRejection::InvalidGrid);
    }
    // Check homogeneous BCs so bilinear Parallelogram Law holds identically
    for bc in &base_spec.face_bcs {
        let zero_bc = match *bc {
            ThermalBcKind::Dirichlet { temp_num, .. } => temp_num == 0,
            ThermalBcKind::Neumann { flux_num, .. } => flux_num == 0,
            ThermalBcKind::Robin { t_inf_num, .. } => t_inf_num == 0,
        };
        if !zero_bc {
            return Err(ThermalRejection::InvalidMaterialParams);
        }
    }

    // Require dual_plus.spec and dual_minus.spec to match base_spec except Q_hot +- alpha / V_cell.
    let gd = BigUint::from_u64(base_spec.grid_den);
    let vol_cell = BigRat::new(
        BigInt::from_i128((base_spec.dx_num as i128) * (base_spec.dy_num as i128) * (base_spec.dz_num as i128)),
        gd.mul_u(&gd).mul_u(&gd),
    );
    let alpha = BigRat::from_i64(alpha_num, alpha_den);
    let q_adj_val = alpha.div(&vol_cell);
    let hot_cid = (hi * base_spec.ny + hj) * base_spec.nz + hk;

    let check_perturbed_spec = |ps: &ThermalSpec3D, sign: i64| -> Result<(), ThermalRejection> {
        if ps.nx != base_spec.nx
            || ps.ny != base_spec.ny
            || ps.nz != base_spec.nz
            || ps.grid_den != base_spec.grid_den
            || ps.dx_num != base_spec.dx_num
            || ps.dy_num != base_spec.dy_num
            || ps.dz_num != base_spec.dz_num
            || ps.material_corners != base_spec.material_corners
            || ps.face_bcs != base_spec.face_bcs
        {
            return Err(ThermalRejection::InvalidGrid);
        }
        if ps.q_src_den == 0 || ps.q_src_num.len() != base_spec.q_src_num.len() {
            return Err(ThermalRejection::SourceSizeMismatch);
        }
        for cid in 0..base_spec.q_src_num.len() {
            let q_base = BigRat::from_i64(base_spec.q_src_num[cid], base_spec.q_src_den);
            let q_exp = if cid == hot_cid {
                if sign > 0 { q_base.add(&q_adj_val) } else { q_base.sub(&q_adj_val) }
            } else {
                q_base
            };
            let q_got = BigRat::from_i64(ps.q_src_num[cid], ps.q_src_den);
            if q_got != q_exp {
                return Err(ThermalRejection::SourceSizeMismatch);
            }
        }
        Ok(())
    };
    check_perturbed_spec(&dual_plus.spec, 1)?;
    check_perturbed_spec(&dual_minus.spec, -1)?;

    let rep_dp = verify_thermal_dual_cert(dual_plus)?;
    let rep_pp = verify_thermal_primal_q1(&dual_plus.spec, primal_plus)?;
    let rep_dm = verify_thermal_dual_cert(dual_minus)?;
    let rep_pm = verify_thermal_primal_q1(&dual_minus.spec, primal_minus)?;

    let four_alpha = BigRat::from_i64(4, 1).mul(&alpha);
    let upper_bound = rep_dp.compliance_upper_per_corner[corner_idx]
        .sub(&rep_pm.compliance_lower_per_corner[corner_idx])
        .div(&four_alpha);
    let lower_bound = rep_pp.compliance_lower_per_corner[corner_idx]
        .sub(&rep_dm.compliance_upper_per_corner[corner_idx])
        .div(&four_alpha);

    Ok(HotSpotTempBracket {
        region_volume: vol_cell,
        lower_bound,
        upper_bound,
    })
}
