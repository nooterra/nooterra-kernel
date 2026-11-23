//! Standalone `#![no_std]`, `#![forbid(unsafe_code)]`, zero-dependency proof verifier
//! for Nooterra 3D continuum multiphysics, geometry, tolerance, and assembly certificates.
//!
//! Core components:
//! - Arbitrary-precision `BigUint`, `BigInt`, and `BigRat` arithmetic (Knuth Algorithm D).
//! - Two-tier PSD verification: Rump (2006) / Higham Thm 10.14 verified shifted Cholesky
//!   with exact `BigInt` fraction-free Bareiss `LDL^T` fallback.
//! - Independent verification of simplicial 3D statics, `HMZ1` contact and yield, Q1 buckling
//!   LMIs, `RT_0` thermal conduction, `ND_0` magnetostatics, curved B-Rep sandwiching, DFM/DFA,
//!   SDT tolerance chains, and `.ntcert` assembly bundles.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod assembly;
pub mod bigint;
pub mod buckling;
pub mod cert;
pub mod check;
pub mod geometry;
pub mod hmz;
pub mod maxwell;
pub mod psd;
pub mod sha256;
pub mod thermal;

pub use assembly::{
    decode_ntcert, encode_ntcert, verify_actuator_assembly_cert, ActuatorAssemblyCert,
    ActuatorAssemblyProofReport, ActuatorTargetSpec, AssemblyRejection, CotsContractsSpec,
    PhysicalScalingContract, ASSEMBLY_KERNEL_ID, NTCERT_MAGIC, NTCERT_VERSION,
};
pub use bigint::{BigInt, BigRat, BigUint};
pub use buckling::{
    exact_q1_ge_hat, exact_q1_ke_hat, verify_hmz_3d_slip_infeasibility, verify_hmz_buckling_lmi,
    BucklingProofReport, BucklingRejection, SlipInfeasibilityReport, SlipRejection,
};
pub use cert::{
    BoundaryFaceKind, BoundaryFaceSpec, MaterialCorner, SimplicialCert3D, Spec3D, Witness3D, G_IDX,
    H_IDX, STRAIN_PAIR,
};
pub use check::{
    verify_cert, verify_cert_with_executor, verify_composed, verify_tet_batch, ComposedProofReport,
    ProofReport, Rejection,
};
pub use geometry::{
    verify_curved_annulus_sandwich, verify_dfm_and_coolant_escape, verify_rigid_assembly_sequence,
    verify_sdt_tolerance_stackup, AssemblyKinematicsReport, AssemblyKinematicsSpec,
    AssemblyPartGeom, BoltHole2D, CurvedAnnulusSpec, CurvedSandwichReport, CurvedSandwichWitness,
    DfmReport, DfmVoxelSpec, DisassemblyStep, GeometryRejection, IntBox3D, SdtToleranceReport,
    SdtToleranceStackupSpec, ToleranceLinkUm, GEOMETRY_KERNEL_ID, VOXEL_AIR, VOXEL_COOLANT,
    VOXEL_PORT, VOXEL_SOLID,
};
pub use hmz::{
    verify_composed_hmz_x, verify_hmz_cert, HmzCert3D, HmzComposedReport, HmzProofReport,
    HmzRejection, HmzSpec3D, HmzWitness3D, HMZ_KERNEL_ID,
};
pub use maxwell::{
    verify_coupled_em_thermal, verify_coupled_em_thermal_gdt, verify_maxwell_dual_cert,
    verify_maxwell_dual_cert_gdt, verify_maxwell_primal_cert, verify_maxwell_primal_cert_gdt,
    verify_reluctance_torque_bracket, verify_reluctance_torque_bracket_gdt,
    CoupledEmThermalReport, MaxwellDualCert3D, MaxwellDualReport, MaxwellMaterialCorner,
    MaxwellPrimalCert3D, MaxwellPrimalReport, MaxwellRejection, MaxwellSpec3D, NedelecWitness3D,
    ReluctanceTorqueBracket, MAXWELL_KERNEL_ID,
};
pub use psd::{verify_psd_bigrat, verify_psd_i128_scaled, PsdMode, PsdOutcome};
pub use thermal::{
    verify_hotspot_temp_bracket, verify_thermal_dual_cert, verify_thermal_dual_cert_gdt,
    verify_thermal_primal_q1, GdtScaleBox3D, HotSpotTempBracket, ThermalBcKind, ThermalDualCert3D,
    ThermalDualReport, ThermalFluxWitness3D, ThermalMaterialCorner, ThermalPrimalReport,
    ThermalPrimalWitness3D, ThermalRejection, ThermalSpec3D, THERMAL_KERNEL_ID,
};

/// Kernel identity baked into every SHA-256 specification hash.
pub const KERNEL_ID: &str = "nooterra-kernel/2.0.0 tet3d-riccati10-twotier";

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn two_tier_agrees_with_exact_bareiss_on_all_regimes() {
        let pd = vec![2i128, -1, 0, -1, 2, -1, 0, -1, 2];
        assert_eq!(verify_psd_i128_scaled(&pd, 3, PsdMode::TwoTierSound), PsdOutcome::ProvenFastRump);
        assert_eq!(verify_psd_i128_scaled(&pd, 3, PsdMode::StrictExactOnly), PsdOutcome::ProvenExactBareiss);

        let psd_zero = vec![1i128, 0, 1, 0, 0, 0, 1, 0, 1];
        assert_eq!(verify_psd_i128_scaled(&psd_zero, 3, PsdMode::TwoTierSound), PsdOutcome::ProvenExactBareiss);
        assert_eq!(verify_psd_i128_scaled(&psd_zero, 3, PsdMode::StrictExactOnly), PsdOutcome::ProvenExactBareiss);

        // Relative det = 1e-14: Tier 1 declines conservatively, Tier 2 certifies exact PSD.
        let barely_pd = vec![1_000_000_000_000_000i128, 1_000_000_000_000_000, 1_000_000_000_000_000, 1_000_000_000_000_001];
        assert_eq!(verify_psd_i128_scaled(&barely_pd, 2, PsdMode::TwoTierSound), PsdOutcome::ProvenExactBareiss);

        // Relative det = -1e-15: both tiers reject.
        let barely_bad = vec![1_000_000_000_000_000i128, 1_000_000_000_000_000, 1_000_000_000_000_000, 999_999_999_999_999];
        assert_eq!(verify_psd_i128_scaled(&barely_bad, 2, PsdMode::TwoTierSound), PsdOutcome::Rejected);
        assert_eq!(verify_psd_i128_scaled(&barely_bad, 2, PsdMode::StrictExactOnly), PsdOutcome::Rejected);

        let bad_zero = vec![1i128, 0, 0, 0, 0, 1, 0, 1, 1];
        assert_eq!(verify_psd_i128_scaled(&bad_zero, 3, PsdMode::TwoTierSound), PsdOutcome::Rejected);

        let asym = vec![2i128, 1, 0, 2];
        assert_eq!(verify_psd_i128_scaled(&asym, 2, PsdMode::TwoTierSound), PsdOutcome::Rejected);
    }

    #[test]
    fn bigrat_to_f64_and_knuth_d_exact() {
        let r = BigRat::from_i64(-355, 113);
        let f = r.to_f64_checked().unwrap();
        assert!((f - (-355.0 / 113.0)).abs() < 1e-14);
        let sum = BigRat::from_i64(1, 3).add(&BigRat::from_i64(1, 6));
        assert_eq!(sum, BigRat::from_i64(1, 2));
    }

    #[test]
    fn hmz1_exact_uniaxial_tension_yield_and_tamper_refusal() {
        // Unit cube [0,1]^3, E = 1, nu = 1/4 (mu = lam = 2/5), uniaxial t_x = 1 on x = 1.
        let free = BoundaryFaceKind::Neumann { traction_num: [0, 0, 0], traction_den: 1 };
        let spec = HmzSpec3D {
            nx: 2, ny: 2, nz: 2, grid_den: 2, dx_num: 1, dy_num: 1, dz_num: 1,
            mat_den: 10,
            material_corners: vec![
                MaterialCorner { mu_num: 4, lam_num: 4, omega2_rho_num: 0 },
            ],
            face_kinds: [
                BoundaryFaceKind::Dirichlet,
                BoundaryFaceKind::Neumann { traction_num: [1, 0, 0], traction_den: 1 },
                free, free, free, free,
            ],
            yield_limit: Some((1, 1)),
        };
        let witness = HmzWitness3D {
            flux_den: 1,
            fx: vec![1; 3 * 2 * 2],
            fy: vec![0; 2 * 3 * 2],
            fz: vec![0; 2 * 2 * 3],
            ex: vec![0; 2 * 3 * 3],
            ey: vec![0; 3 * 2 * 3],
            ez: vec![0; 3 * 3 * 2],
        };
        let cert = HmzCert3D { spec, witness };
        let rep = verify_hmz_cert(&cert).expect("exact uniaxial HMZ1 must pass");
        assert_eq!(rep.worst_compliance_upper, BigRat::from_i64(1, 1));
        assert_eq!(rep.max_von_mises_sq, BigRat::from_i64(1, 1));

        let mut bad_y = cert.clone();
        bad_y.spec.yield_limit = Some((99, 100));
        assert!(matches!(verify_hmz_cert(&bad_y), Err(HmzRejection::YieldLimitExceeded { .. })));

        let mut bad = cert.clone();
        bad.witness.fx[hmz::idx_fx(2, 2, 1, 0, 0)] += 1;
        assert!(matches!(verify_hmz_cert(&bad), Err(HmzRejection::EquilibriumViolation { .. })));
    }

    #[test]
    fn hmz1_3d_circular_friction_cone_and_farkas_slip_infeasibility() {
        // Unit cube with contact (mu = 0.30, gap = 0.01) on x = 0; p_n = 1.00, tau = (-0.18, -0.24).
        let make_spec = |ty_num: i64, tz_num: i64, gap_num: i64| HmzSpec3D {
            nx: 1, ny: 1, nz: 1, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            mat_den: 10,
            material_corners: vec![MaterialCorner { mu_num: 4, lam_num: 4, omega2_rho_num: 0 }],
            face_kinds: [
                BoundaryFaceKind::Contact { mu_num: 3, mu_den: 10, gap_num, gap_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [-100, ty_num, tz_num], traction_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [-ty_num, 0, 0], traction_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [ty_num, 0, 0], traction_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [-tz_num, 0, 0], traction_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [tz_num, 0, 0], traction_den: 100 },
            ],
            yield_limit: None,
        };
        let safe_cert = HmzCert3D {
            spec: make_spec(-18, -24, 1),
            witness: HmzWitness3D {
                flux_den: 100,
                fx: vec![-100; 2],
                fy: vec![0; 2],
                fz: vec![0; 2],
                ex: vec![0; 4],
                ey: vec![-24; 4],
                ez: vec![-18; 4],
            },
        };
        let rep = verify_hmz_cert(&safe_cert).expect("Pythagorean boundary shear (0.18, 0.24) <= 0.30 must pass");
        assert_eq!(rep.contact_faces_verified, 1);
        assert_eq!(rep.contact_gap_compliance, BigRat::from_i64(1, 50));

        let mut slip_cert = safe_cert.clone();
        slip_cert.spec = make_spec(-19, -24, 0);
        slip_cert.witness.ez = vec![-19; 4];
        assert!(matches!(
            verify_hmz_cert(&slip_cert),
            Err(HmzRejection::ContactCircularFrictionViolation { side: 0, j: 0, k: 0 })
        ));

        let mut tens_cert = safe_cert.clone();
        tens_cert.spec.face_kinds[1] = BoundaryFaceKind::Neumann { traction_num: [1, -18, -24], traction_den: 100 };
        tens_cert.witness.fx = vec![1; 2];
        assert!(matches!(
            verify_hmz_cert(&tens_cert),
            Err(HmzRejection::ContactTensionViolation { side: 0, j: 0, k: 0 })
        ));

        // Diagonal shear (-0.21, -0.28) passes per-axis checks (< 0.30) but exceeds circular cone (0.35 > 0.30).
        let unsafe_spec = HmzSpec3D {
            nx: 1, ny: 1, nz: 1, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            mat_den: 10,
            material_corners: vec![MaterialCorner { mu_num: 4, lam_num: 4, omega2_rho_num: 0 }],
            face_kinds: [
                BoundaryFaceKind::Contact { mu_num: 3, mu_den: 10, gap_num: 0, gap_den: 1 },
                BoundaryFaceKind::Neumann { traction_num: [-100, 0, 0], traction_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [0, 0, 0], traction_den: 1 },
                BoundaryFaceKind::Neumann { traction_num: [0, -21, 0], traction_den: 100 },
                BoundaryFaceKind::Neumann { traction_num: [0, 0, 0], traction_den: 1 },
                BoundaryFaceKind::Neumann { traction_num: [0, 0, -28], traction_den: 100 },
            ],
            yield_limit: None,
        };
        let srep = verify_hmz_3d_slip_infeasibility(&unsafe_spec, [30, -60, -80])
            .expect("Pythagorean slip witness (30, -60, -80) must prove 3D infeasibility");
        assert_eq!(srep.virtual_work, BigRat::from_i64(5, 1));
        let mut safe_f_spec = unsafe_spec.clone();
        safe_f_spec.face_kinds[3] = BoundaryFaceKind::Neumann { traction_num: [0, -18, 0], traction_den: 100 };
        safe_f_spec.face_kinds[5] = BoundaryFaceKind::Neumann { traction_num: [0, 0, -24], traction_den: 100 };
        assert_eq!(
            verify_hmz_3d_slip_infeasibility(&safe_f_spec, [30, -60, -80]),
            Err(SlipRejection::NonPositiveVirtualWork)
        );
    }

    #[test]
    fn buckling_exact_q1_integrals_match_gauss_and_certify_column() {
        let (dx_i, dy_i, dz_i) = (3i128, 2i128, 5i128);
        let (dx, dy, dz) = (3.0f64, 2.0f64, 5.0f64);
        let (mu_i, lam_i) = (4i128, 6i128);
        let (mu, lam) = (4.0f64, 6.0f64);
        let p = dx_i * dy_i * dz_i;
        let scale = 3456.0 * (p as f64) * (p as f64);

        let ke_exact = exact_q1_ke_hat(dx_i, dy_i, dz_i, mu_i, lam_i).unwrap();

        let (fx0, fx1, fy0, fy1, fz0, fz1) = (-7i128, -3i128, 2i128, -1i128, 4i128, 1i128);
        let ez = [2i128, -3, 5, -1];
        let ey = [-4i128, 1, -2, 6];
        let ex = [3i128, 2, -5, 4];
        let dxy = dx_i * dy_i;
        let dxz = dx_i * dz_i;
        let dyz = dy_i * dz_i;
        let tw_z = ez[0] - ez[1] - ez[2] + ez[3];
        let tw_y = ey[0] - ey[1] - ey[2] + ey[3];
        let tw_x = ex[0] - ex[1] - ex[2] + ex[3];
        let qx = (-3 * dx_i * (dxz * tw_z + dxy * tw_y)) as f64 / (24.0 * p as f64);
        let qy = (-3 * dy_i * (dyz * tw_z + dxy * tw_x)) as f64 / (24.0 * p as f64);
        let qz = (-3 * dz_i * (dyz * tw_y + dxz * tw_x)) as f64 / (24.0 * p as f64);
        let sbx_l = 12 * p * (fx0 + fx1) + 2 * dx_i * (dxz * tw_z + dxy * tw_y);
        let sby_l = 12 * p * (fy0 + fy1) + 2 * dy_i * (dyz * tw_z + dxy * tw_x);
        let sbz_l = 12 * p * (fz0 + fz1) + 2 * dz_i * (dyz * tw_y + dxz * tw_x);

        let ge_exact = exact_q1_ge_hat(dx_i, dy_i, dz_i, sbx_l, sby_l, sbz_l, ez, ey, ex).unwrap();

        // 3x3x3 Gauss-Legendre quadrature is exact up to polynomial degree 5.
        let gp = [-(0.6f64).sqrt(), 0.0, (0.6f64).sqrt()];
        let gw = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0];
        let mut ke_num = [[0.0f64; 24]; 24];
        let mut ge_num = [[0.0f64; 8]; 8];
        let det_j = 0.125 * dx * dy * dz;

        for ix in 0..3 {
            let xi = gp[ix];
            for iy in 0..3 {
                let eta = gp[iy];
                for iz in 0..3 {
                    let zeta = gp[iz];
                    let w = gw[ix] * gw[iy] * gw[iz] * det_j;
                    let mut dnx = [0.0f64; 8];
                    let mut dny = [0.0f64; 8];
                    let mut dnz = [0.0f64; 8];
                    for a in 0..8 {
                        let sx = if (a >> 2) == 0 { -1.0 } else { 1.0 };
                        let sy = if ((a >> 1) & 1) == 0 { -1.0 } else { 1.0 };
                        let sz = if (a & 1) == 0 { -1.0 } else { 1.0 };
                        dnx[a] = (sx / dx) * 0.5 * (1.0 + sy * eta) * 0.5 * (1.0 + sz * zeta);
                        dny[a] = 0.5 * (1.0 + sx * xi) * (sy / dy) * 0.5 * (1.0 + sz * zeta);
                        dnz[a] = 0.5 * (1.0 + sx * xi) * 0.5 * (1.0 + sy * eta) * (sz / dz);
                    }
                    let mut bmat = [[0.0f64; 24]; 6];
                    for a in 0..8 {
                        bmat[0][3 * a] = dnx[a];
                        bmat[1][3 * a + 1] = dny[a];
                        bmat[2][3 * a + 2] = dnz[a];
                        bmat[3][3 * a] = dny[a];
                        bmat[3][3 * a + 1] = dnx[a];
                        bmat[4][3 * a] = dnz[a];
                        bmat[4][3 * a + 2] = dnx[a];
                        bmat[5][3 * a + 1] = dnz[a];
                        bmat[5][3 * a + 2] = dny[a];
                    }
                    for r in 0..24 {
                        let tr = bmat[0][r] + bmat[1][r] + bmat[2][r];
                        let cr = [
                            2.0 * mu * bmat[0][r] + lam * tr,
                            2.0 * mu * bmat[1][r] + lam * tr,
                            2.0 * mu * bmat[2][r] + lam * tr,
                            mu * bmat[3][r],
                            mu * bmat[4][r],
                            mu * bmat[5][r],
                        ];
                        for c in 0..24 {
                            let mut s = 0.0;
                            for k in 0..6 {
                                s += cr[k] * bmat[k][c];
                            }
                            ke_num[r][c] += w * s;
                        }
                    }
                    let sxx = 0.5 * (fx0 + fx1) as f64 + 0.5 * (fx1 - fx0) as f64 * xi + qx * (xi * xi - 1.0);
                    let syy = 0.5 * (fy0 + fy1) as f64 + 0.5 * (fy1 - fy0) as f64 * eta + qy * (eta * eta - 1.0);
                    let szz = 0.5 * (fz0 + fz1) as f64 + 0.5 * (fz1 - fz0) as f64 * zeta + qz * (zeta * zeta - 1.0);
                    let bil = |e: [i128; 4], u: f64, v: f64| -> f64 {
                        0.25 * (e[0] as f64 * (1.0 - u) * (1.0 - v)
                            + e[1] as f64 * (1.0 + u) * (1.0 - v)
                            + e[2] as f64 * (1.0 - u) * (1.0 + v)
                            + e[3] as f64 * (1.0 + u) * (1.0 + v))
                    };
                    let sxy = bil(ez, xi, eta);
                    let sxz = bil(ey, xi, zeta);
                    let syz = bil(ex, eta, zeta);
                    for a in 0..8 {
                        for b in 0..8 {
                            let g = dnx[a] * (sxx * dnx[b] + sxy * dny[b] + sxz * dnz[b])
                                + dny[a] * (sxy * dnx[b] + syy * dny[b] + syz * dnz[b])
                                + dnz[a] * (sxz * dnx[b] + syz * dny[b] + szz * dnz[b]);
                            ge_num[a][b] += w * g;
                        }
                    }
                }
            }
        }
        for r in 0..24 {
            for c in 0..24 {
                let ex_v = ke_exact[r][c] as f64 / scale;
                assert!((ex_v - ke_num[r][c]).abs() < 1e-11, "Ke mismatch at ({r},{c}): {ex_v} vs {}", ke_num[r][c]);
            }
        }
        for a in 0..8 {
            for b in 0..8 {
                let ex_v = ge_exact[a][b] as f64 / scale;
                assert!((ex_v - ge_num[a][b]).abs() < 1e-11, "Ge mismatch at ({a},{b}): {ex_v} vs {}", ge_num[a][b]);
            }
        }

        // Cantilever [0,4]x[0,1]x[0,1], E = 1, sigma_xx = -0.01 has Euler P_cr ~ 0.01285 (Lambda_crit ~ 1.285).
        let free = BoundaryFaceKind::Neumann { traction_num: [0, 0, 0], traction_den: 1 };
        let spec = HmzSpec3D {
            nx: 4, ny: 1, nz: 1, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            mat_den: 10,
            material_corners: vec![MaterialCorner { mu_num: 4, lam_num: 4, omega2_rho_num: 0 }],
            face_kinds: [
                BoundaryFaceKind::Dirichlet,
                BoundaryFaceKind::Neumann { traction_num: [-1, 0, 0], traction_den: 100 },
                free, free, free, free,
            ],
            yield_limit: Some((1, 50)),
        };
        let witness = HmzWitness3D {
            flux_den: 100,
            fx: vec![-1; 5],
            fy: vec![0; 8],
            fz: vec![0; 8],
            ex: vec![0; 16],
            ey: vec![0; 10],
            ez: vec![0; 10],
        };
        let cert = HmzCert3D { spec, witness };
        let rep_ok = verify_hmz_buckling_lmi(&cert, 1, 1, PsdMode::TwoTierSound)
            .expect("sub-critical Lambda = 1.0 must pass");
        assert_eq!(rep_ok.corner_outcomes[0], PsdOutcome::ProvenFastRump);
        assert_eq!(rep_ok.n_free_dofs, 48);

        let err_bad = verify_hmz_buckling_lmi(&cert, 2, 1, PsdMode::TwoTierSound).unwrap_err();
        assert_eq!(err_bad, BucklingRejection::BucklingLmiFailed { corner: 0 });
    }

    #[test]
    fn thermal_cht_robin_exact_zero_duality_gap_and_tamper() {
        // 1D CHT on [0,1]^3 with k = 1, Q = 0, T(0) = 0, Robin h = 1, T_inf = 1 at x = 1.
        // Analytic solution: T(x) = x/2, q_x = -1/2, primal and dual bounds both equal -1/2.
        let adiab = ThermalBcKind::Neumann { flux_num: 0, flux_den: 1 };
        let spec = ThermalSpec3D {
            nx: 2, ny: 1, nz: 1, grid_den: 2, dx_num: 1, dy_num: 2, dz_num: 2,
            q_src_den: 1,
            q_src_num: vec![0, 0],
            material_corners: vec![ThermalMaterialCorner {
                mat_den: 1,
                k_num: vec![1, 1],
                robin_h_num: [0, 1, 0, 0, 0, 0],
            }],
            face_bcs: [
                ThermalBcKind::Dirichlet { temp_num: 0, temp_den: 1 },
                ThermalBcKind::Robin { t_inf_num: 1, t_inf_den: 1 },
                adiab, adiab, adiab, adiab,
            ],
        };
        let dual_cert = ThermalDualCert3D {
            spec: spec.clone(),
            witness: ThermalFluxWitness3D {
                flux_den: 2,
                qx: vec![-1, -1, -1],
                qy: vec![0; 4],
                qz: vec![0; 4],
            },
        };
        let rep_d = verify_thermal_dual_cert(&dual_cert).expect("exact Robin CHT dual must pass");
        assert_eq!(rep_d.worst_compliance_upper, BigRat::from_i64(-1, 2));

        let mut t_nodes = vec![0i64; 3 * 2 * 2];
        for ix in 0..=2 {
            for iy in 0..=1 {
                for iz in 0..=1 {
                    t_nodes[thermal::idx_tnode(1, 1, ix, iy, iz)] = ix as i64;
                }
            }
        }
        let rep_p = verify_thermal_primal_q1(&spec, &ThermalPrimalWitness3D { temp_den: 4, t_nodes })
            .expect("exact Robin CHT primal must pass");
        assert_eq!(rep_p.compliance_lower_per_corner[0], BigRat::from_i64(-1, 2));

        let mut bad = dual_cert.clone();
        bad.witness.qx[1] += 1;
        assert!(matches!(verify_thermal_dual_cert(&bad), Err(ThermalRejection::DivergenceViolation { .. })));
    }

    #[test]
    fn maxwell_nedelec_exact_ampere_primal_dual_and_coupled_joule() {
        // Single nonzero edge H_z(1,1,1) = 1 induces closed current loop J = curl(H) on 4 adjacent faces.
        let (nx, ny, nz) = (3, 3, 3);
        let mut hx = vec![0i64; nx * (ny + 1) * (nz + 1)];
        let mut hy = vec![0i64; (nx + 1) * ny * (nz + 1)];
        let mut hz = vec![0i64; (nx + 1) * (ny + 1) * nz];
        hz[maxwell::idx_edge_z(ny, nz, 1, 1, 1)] = 1;
        let _ = (&mut hx, &mut hy);

        let mut jx = vec![0i64; (nx + 1) * ny * nz];
        let mut jy = vec![0i64; nx * (ny + 1) * nz];
        let jz = vec![0i64; nx * ny * (nz + 1)];
        jx[thermal::idx_qx(ny, nz, 1, 0, 1)] = 1;
        jx[thermal::idx_qx(ny, nz, 1, 1, 1)] = -1;
        jy[thermal::idx_qy(ny, nz, 0, 1, 1)] = -1;
        jy[thermal::idx_qy(ny, nz, 1, 1, 1)] = 1;

        let em_spec = MaxwellSpec3D {
            nx, ny, nz, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            j_den: 1, jx, jy, jz,
            material_corners: vec![MaxwellMaterialCorner { mat_den: 1, mu_num: vec![1; 27] }],
        };
        let em_dual = MaxwellDualCert3D {
            spec: em_spec.clone(),
            h_witness: NedelecWitness3D { field_den: 1, ex: hx, ey: hy, ez: hz },
        };
        let rep_d = verify_maxwell_dual_cert(&em_dual).expect("exact Ampere Nedelec dual must pass");
        assert_eq!(rep_d.worst_energy2_upper, BigRat::from_i64(4, 9));

        let em_primal = MaxwellPrimalCert3D {
            spec: em_spec.clone(),
            a_witness: NedelecWitness3D {
                field_den: 1,
                ex: vec![0; nx * (ny + 1) * (nz + 1)],
                ey: vec![0; (nx + 1) * ny * (nz + 1)],
                ez: vec![0; (nx + 1) * (ny + 1) * nz],
            },
        };
        let rep_p = verify_maxwell_primal_cert(&em_primal).expect("valid primal must pass");
        assert!(rep_p.energy2_lower_per_corner[0] <= rep_d.energy2_upper_per_corner[0]);

        let mut bad_em = em_dual.clone();
        bad_em.h_witness.ez[maxwell::idx_edge_z(ny, nz, 1, 1, 1)] += 1;
        assert!(matches!(verify_maxwell_dual_cert(&bad_em), Err(MaxwellRejection::AmpereCurlViolation { .. })));

        // Each of the 4 cells touching the loop carries one x-face and one y-face (|J|^2 integral = 2/3).
        let mut q_src = vec![0i64; 27];
        for &(ci, cj) in &[(0, 0), (1, 0), (0, 1), (1, 1)] {
            q_src[(ci * ny + cj) * nz + 1] = 2;
        }
        let dir0 = ThermalBcKind::Dirichlet { temp_num: 0, temp_den: 1 };
        let th_spec = ThermalSpec3D {
            nx, ny, nz, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            q_src_den: 3, q_src_num: q_src,
            material_corners: vec![ThermalMaterialCorner { mat_den: 1, k_num: vec![1; 27], robin_h_num: [0; 6] }],
            face_bcs: [dir0; 6],
        };
        let mut qx = vec![0i64; (nx + 1) * ny * nz];
        for i in (0..nx).rev() {
            for j in 0..ny {
                for k in 0..nz {
                    let cid = (i * ny + j) * nz + k;
                    qx[thermal::idx_qx(ny, nz, i, j, k)] =
                        qx[thermal::idx_qx(ny, nz, i + 1, j, k)] - th_spec.q_src_num[cid];
                }
            }
        }
        let th_cert = ThermalDualCert3D {
            spec: th_spec,
            witness: ThermalFluxWitness3D {
                flux_den: 3, qx, qy: vec![0; nx * (ny + 1) * nz], qz: vec![0; nx * ny * (nz + 1)],
            },
        };
        let sig = vec![1i64; 27];
        let crep = verify_coupled_em_thermal(&em_dual, &th_cert, &sig, 1)
            .expect("exact Joule coupling must pass");
        assert_eq!(crep.active_conductor_cells, 4);
        assert_eq!(crep.total_joule_power, BigRat::from_i64(8, 3));

        let mut bad_th = th_cert.clone();
        bad_th.spec.q_src_num[(0 * ny + 0) * nz + 1] = 1;
        bad_th.witness.qx[thermal::idx_qx(ny, nz, 0, 0, 1)] += 1;
        assert!(matches!(
            verify_coupled_em_thermal(&em_dual, &bad_th, &sig, 1),
            Err(MaxwellRejection::JouleSourceUnderestimated { i: 0, j: 0, k: 1 })
        ));
    }

    #[test]
    fn gdt_piola_8corner_exactness_for_thermal_and_maxwell() {
        // Q = 2, T(0) = 0, adiabatic at x = 1 gives q_x = -2(1 - x) and nominal energy 4/3.
        // Pure x-flux scales by ax / (ay * az), peaking at (11/10) / (9/10)^2 = 110/81.
        let adiab = ThermalBcKind::Neumann { flux_num: 0, flux_den: 1 };
        let th_spec = ThermalSpec3D {
            nx: 1, ny: 1, nz: 1, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            q_src_den: 1, q_src_num: vec![2],
            material_corners: vec![ThermalMaterialCorner { mat_den: 1, k_num: vec![1], robin_h_num: [0; 6] }],
            face_bcs: [ThermalBcKind::Dirichlet { temp_num: 0, temp_den: 1 }, adiab, adiab, adiab, adiab, adiab],
        };
        let th_cert = ThermalDualCert3D {
            spec: th_spec,
            witness: ThermalFluxWitness3D { flux_den: 1, qx: vec![-2, 0], qy: vec![0, 0], qz: vec![0, 0] },
        };
        let gdt = GdtScaleBox3D {
            scale_den: 10,
            ax_min: 9, ax_max: 11,
            ay_min: 9, ay_max: 11,
            az_min: 9, az_max: 11,
        };
        let r_nom = verify_thermal_dual_cert(&th_cert).unwrap();
        assert_eq!(r_nom.worst_compliance_upper, BigRat::from_i64(4, 3));
        let r_gdt = verify_thermal_dual_cert_gdt(&th_cert, &gdt).unwrap();
        assert_eq!(r_gdt.worst_compliance_upper, BigRat::from_i64(440, 243));
    }

    #[test]
    fn curved_brep_sandwich_dfm_powder_escape_dfa_and_sdt() {
        let spec = CurvedAnnulusSpec {
            nx: 10, ny: 10, nz: 2, grid_den: 1, dx_num: 1, dy_num: 1, dz_num: 1,
            r_den: 2, center_x_num: 10, center_y_num: 10, r_in_num: 4, r_out_num: 8, z_min_num: 0, z_max_num: 4,
            bolt_holes: vec![BoltHole2D { cx_num: 16, cy_num: 10, radius_num: 1 }],
        };
        let n = 10 * 10 * 2;
        let mut in_mask = vec![0u8; n];
        let mut out_mask = vec![0u8; n];
        for i in 0..10 {
            for j in 0..10 {
                let x0 = (2 * i) as i64; let x1 = x0 + 2;
                let y0 = (2 * j) as i64; let y1 = y0 + 2;
                let dx_min = if 10 < x0 { x0 - 10 } else if 10 > x1 { 10 - x1 } else { 0 };
                let dy_min = if 10 < y0 { y0 - 10 } else if 10 > y1 { 10 - y1 } else { 0 };
                let dx_max = (x0 - 10).abs().max((x1 - 10).abs());
                let dy_max = (y0 - 10).abs().max((y1 - 10).abs());
                let rmin2 = dx_min * dx_min + dy_min * dy_min;
                let rmax2 = dx_max * dx_max + dy_max * dy_max;
                let bdx_min = if 16 < x0 { x0 - 16 } else if 16 > x1 { 16 - x1 } else { 0 };
                let bdy_min = if 10 < y0 { y0 - 10 } else if 10 > y1 { 10 - y1 } else { 0 };
                let bmin2 = bdx_min * bdx_min + bdy_min * bdy_min;
                let bdx_max = (x0 - 16).abs().max((x1 - 16).abs());
                let bdy_max = (y0 - 10).abs().max((y1 - 10).abs());
                let bmax2 = bdx_max * bdx_max + bdy_max * bdy_max;
                let strict_in = rmin2 >= 16 && rmax2 <= 64 && bmin2 >= 1;
                let strict_out = rmax2 <= 16 || rmin2 >= 64 || bmax2 <= 1;
                for k in 0..2 {
                    let cid = (i * 10 + j) * 2 + k;
                    if strict_in { in_mask[cid] = 1; }
                    if !strict_out { out_mask[cid] = 1; }
                }
            }
        }
        let w = CurvedSandwichWitness { in_mask: in_mask.clone(), out_mask: out_mask.clone() };
        let gre = verify_curved_annulus_sandwich(&spec, &w).expect("exact curved sandwich must pass");
        assert!(gre.inner_cells > 0 && gre.outer_cells > gre.inner_cells);
        let mut bad_w = w.clone();
        let bnd_cid = (0..n).find(|&c| out_mask[c] == 1 && in_mask[c] == 0).unwrap();
        bad_w.in_mask[bnd_cid] = 1;
        assert!(matches!(
            verify_curved_annulus_sandwich(&spec, &bad_w),
            Err(GeometryRejection::InnerShellProtrudesOutsideBrep { .. })
        ));

        let mut vox = vec![VOXEL_SOLID; 6 * 6 * 4];
        let vid = |i: usize, j: usize, k: usize| (i * 6 + j) * 4 + k;
        vox[vid(0, 2, 1)] = VOXEL_PORT;
        vox[vid(1, 2, 1)] = VOXEL_COOLANT;
        vox[vid(2, 2, 1)] = VOXEL_COOLANT;
        vox[vid(2, 3, 1)] = VOXEL_COOLANT;
        let dfm = DfmVoxelSpec {
            nx: 6, ny: 6, nz: 4, voxels: vox.clone(),
            min_wall_cells: 1, check_45deg_overhang: true, require_powder_escape: true,
        };
        verify_dfm_and_coolant_escape(&dfm).expect("connected coolant channel must pass");
        let mut bad_dfm = dfm.clone();
        bad_dfm.voxels[vid(4, 4, 2)] = VOXEL_COOLANT;
        assert_eq!(
            verify_dfm_and_coolant_escape(&bad_dfm),
            Err(GeometryRejection::TrappedCoolantPowder { i: 4, j: 4, k: 2 })
        );

        let housing = AssemblyPartGeom {
            part_id: 1,
            boxes: vec![
                IntBox3D { x0: 0, x1: 10, y0: 0, y1: 10, z0: 0, z1: 2 },
                IntBox3D { x0: 0, x1: 2, y0: 0, y1: 10, z0: 2, z1: 8 },
                IntBox3D { x0: 8, x1: 10, y0: 0, y1: 10, z0: 2, z1: 8 },
            ],
        };
        let stator = AssemblyPartGeom {
            part_id: 2,
            boxes: vec![IntBox3D { x0: 2, x1: 8, y0: 2, y1: 8, z0: 2, z1: 7 }],
        };
        let cap = AssemblyPartGeom {
            part_id: 3,
            boxes: vec![IntBox3D { x0: 0, x1: 10, y0: 0, y1: 10, z0: 8, z1: 10 }],
        };
        let dfa_ok = AssemblyKinematicsSpec {
            parts: vec![housing.clone(), stator.clone(), cap.clone()],
            disassembly_steps: vec![
                DisassemblyStep { part_id: 3, axis: 2, dir: 1 },
                DisassemblyStep { part_id: 2, axis: 2, dir: 1 },
                DisassemblyStep { part_id: 1, axis: 2, dir: -1 },
            ],
        };
        verify_rigid_assembly_sequence(&dfa_ok).expect("valid disassembly must pass");
        let mut dfa_bad = dfa_ok.clone();
        dfa_bad.disassembly_steps.swap(0, 1);
        assert_eq!(
            verify_rigid_assembly_sequence(&dfa_bad),
            Err(GeometryRejection::AssemblyCollision { moving_part: 2, blocking_part: 3, step_idx: 0 })
        );

        let sdt = SdtToleranceStackupSpec {
            nominal_x_um: 100_000, nominal_y_um: 100_000, nominal_z_um: 50_000,
            nominal_airgap_um: 500, min_safe_airgap_um: 400, flange_scale_um: 50_000,
            links: vec![
                ToleranceLinkUm { link_id: 1, pm_x_um: 20, pm_y_um: 20, pm_z_um: 15, pm_airgap_um: 12, pm_flange_gap_um: 15 },
                ToleranceLinkUm { link_id: 2, pm_x_um: 10, pm_y_um: 10, pm_z_um: 10, pm_airgap_um: 8, pm_flange_gap_um: 10 },
            ],
        };
        let srep = verify_sdt_tolerance_stackup(&sdt).expect("safe tolerance chain must pass");
        assert_eq!(srep.min_airgap_um, 480);
        assert_eq!(srep.worst_flange_gap_norm, BigRat::from_i64(25, 50_000));
        let mut bad_sdt = sdt.clone();
        bad_sdt.links[0].pm_airgap_um = 100;
        assert_eq!(
            verify_sdt_tolerance_stackup(&bad_sdt),
            Err(GeometryRejection::AirgapRubRisk { min_airgap_num: 392, required_min_num: 400 })
        );
    }
}
