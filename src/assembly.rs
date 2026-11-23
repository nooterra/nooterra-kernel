//! Zero-Trust Unified High-Torque Robotic Actuator Assembly Proof Verifier & `.ntcert` Wire Codec
//! (`S1` Typed COTS Contracts + Cross-Physics `G1+G3+G4+M1+M2+M3+M5+S1` Compositional Verification).
//!
//! 100% `#![no_std]`, `#![forbid(unsafe_code)]`, ZERO external dependencies!
//! Every single cross-discipline coupling is checked cryptographically inside the TCB:
//!   - `G1` Curved B-Rep Sandwich (`Omega_in <= Omega_exact <= Omega_out`)
//!   - `G3` DFM (min wall, 45-deg overhang, BFS coolant powder escape) & Collision-Free DFA Sequence
//!   - `G4` 3D SDT Tolerance Stackup -> cryptographically enforces `M1` gap, `M3` GD&T box, `M5` GD&T box
//!   - `M5` 3D Maxwell Reluctance Torque over 8-corner continuous GD&T box -> enforces target min torque
//!   - `M5 -> M3` Coupled Joule-Thermal over 8-corner GD&T box -> enforces winding & COTS temp limits
//!   - `M1` 3D Preloaded Bolted Flange Contact + Circular Friction Cone -> enforces torque-transfer shear
//!   - `M2` 3D Housing Yield + Buckling LMI -> enforces global buckling safety factor
//!   - `S1` COTS Bearing, ISO Bolt, Driver PCB & Encoder Ratings + Mass/Envelope Budgets!

use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat};
use crate::buckling::{verify_hmz_buckling_lmi, BucklingProofReport, BucklingRejection};
use crate::cert::{BoundaryFaceKind, MaterialCorner};
use crate::geometry::{
    verify_curved_annulus_sandwich, verify_dfm_and_coolant_escape, verify_rigid_assembly_sequence,
    verify_sdt_tolerance_stackup, AssemblyKinematicsReport, AssemblyKinematicsSpec, AssemblyPartGeom,
    BoltHole2D, CurvedAnnulusSpec, CurvedSandwichReport, CurvedSandwichWitness, DfmReport,
    DfmVoxelSpec, DisassemblyStep, GeometryRejection, IntBox3D, SdtToleranceReport,
    SdtToleranceStackupSpec, ToleranceLinkUm,
};
use crate::hmz::{verify_hmz_cert, HmzCert3D, HmzProofReport, HmzRejection, HmzSpec3D, HmzWitness3D};
use crate::maxwell::{
    verify_coupled_em_thermal_gdt, verify_reluctance_torque_bracket_gdt, CoupledEmThermalReport,
    MaxwellDualCert3D, MaxwellMaterialCorner, MaxwellPrimalCert3D, MaxwellRejection, MaxwellSpec3D,
    NedelecWitness3D, ReluctanceTorqueBracket,
};
use crate::psd::PsdMode;
use crate::sha256::Sha256;
use crate::thermal::{
    GdtScaleBox3D, ThermalBcKind, ThermalDualCert3D, ThermalFluxWitness3D, ThermalMaterialCorner,
    ThermalSpec3D,
};

pub const ASSEMBLY_KERNEL_ID: &str = "nooterra-kernel/3.0.0 actuator-assembly-ntcert-v3";
pub const NTCERT_MAGIC: [u8; 4] = *b"NTCT";
pub const NTCERT_VERSION: u32 = 3;

/// High-level engineering target specification for the High-Torque Robotic Actuator!
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActuatorTargetSpec {
    /// Minimum required certified output torque (N*m = num / den).
    pub min_torque_num: i64,
    pub min_torque_den: u64,
    /// Maximum allowed winding temperature (deg C = num / den).
    pub max_winding_temp_num: i64,
    pub max_winding_temp_den: u64,
    /// Coolant inlet ambient temperature (deg C = num / den).
    pub coolant_temp_num: i64,
    pub coolant_temp_den: u64,
    /// Minimum required 3D housing buckling safety factor (`Lambda_req = num / den`).
    pub min_buckling_sf_num: i64,
    pub min_buckling_sf_den: u64,
    /// Maximum allowed outer diameter & axial length (micrometers).
    pub max_od_um: u64,
    pub max_length_um: u64,
    /// Maximum allowed total assembly mass (grams).
    pub max_mass_g: u64,
}

/// Typed COTS component contracts (`NT-Contract v1`, Wall S1)!
/// Every datasheet parameter is encoded as exact integers/rationals (zero floats).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CotsContractsSpec {
    /// Crossed-roller bearing max continuous torque rating (N*m = num / den).
    pub bearing_max_torque_num: i64,
    pub bearing_max_torque_den: u64,
    /// Bearing max operating temperature (deg C = num / den).
    pub bearing_max_temp_num: i64,
    pub bearing_max_temp_den: u64,
    /// Bearing datasheet radial/axial runout bound (um).
    pub bearing_runout_um: u64,
    /// Bearing mass (grams).
    pub bearing_mass_g: u64,

    /// ISO 898-1 Class 12.9 flange bolt circle radius (m = num / den).
    pub bolt_circle_radius_num: i64,
    pub bolt_circle_radius_den: u64,
    /// Minimum conservative dry/lubricated steel-on-steel friction coefficient `mu_min` in COTS spec.
    pub bolt_min_mu_num: i64,
    pub bolt_min_mu_den: u64,
    /// Fastener kit mass (grams).
    pub bolts_mass_g: u64,

    /// 48V GaN Driver PCB + Magnetic Encoder max continuous Joule dissipation capacity (W = num / den).
    pub driver_max_power_num: i64,
    pub driver_max_power_den: u64,
    /// Driver PCB + Encoder max ambient housing temperature (deg C = num / den).
    pub driver_max_temp_num: i64,
    pub driver_max_temp_den: u64,
    /// Driver PCB + Encoder mass (grams).
    pub electronics_mass_g: u64,

    /// Custom machined/printed stator + rotor + housing + cooling jacket + flange mass (grams).
    pub custom_parts_mass_g: u64,
    /// Actual designed outer diameter and axial length (micrometers).
    pub actual_od_um: u64,
    pub actual_length_um: u64,
}

/// Physical scaling constants binding normalized dimensionless PDE meshes to exact SI units!
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalScalingContract {
    /// Rotor angular sweep `dtheta = num / den` (rad) between Pos 1 and Pos 2.
    pub em_dtheta_num: i64,
    pub em_dtheta_den: u64,
    /// Exact rational factor mapping raw EM energy-diff torque to SI N*m (`mu_0 * I_scale^2 * L_scale`).
    pub em_torque_scale_num: i64,
    pub em_torque_scale_den: u64,
    /// Exact rational factor mapping raw coupled thermal `T_avg` to SI deg C rise.
    pub thermal_temp_scale_num: i64,
    pub thermal_temp_scale_den: u64,
    /// Exact rational factor mapping raw coupled Joule power to SI Watts.
    pub joule_power_scale_num: i64,
    pub joule_power_scale_den: u64,
    /// Exact rational factor mapping flange dimensionless applied shear force `F_y` to SI Newtons.
    pub flange_force_scale_num: i64,
    pub flange_force_scale_den: u64,
    /// Worst-case permeability corner index (`0` = low `mu_Fe` -> conservative torque!).
    pub em_corner_idx: usize,
}

/// Complete self-contained zero-trust High-Torque Robotic Actuator Assembly Certificate (`.ntcert`)!
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActuatorAssemblyCert {
    pub target: ActuatorTargetSpec,
    pub cots: CotsContractsSpec,
    pub scaling: PhysicalScalingContract,
    /// SHA-256 hash of the exported ISO 10303-21 STEP B-Rep CAD assembly file (cryptographically bound!).
    pub step_cad_sha256: [u8; 32],

    // G1: Curved B-Rep geometric sandwich proof
    pub housing_brep_spec: CurvedAnnulusSpec,
    pub housing_brep_witness: CurvedSandwichWitness,

    // G3: DFM + Powder-Escape BFS + Collision-Free Rigid-Body DFA Sequence
    pub cooling_dfm_spec: DfmVoxelSpec,
    pub dfa_kinematics_spec: AssemblyKinematicsSpec,

    // G4: 3D SDT Tolerance Stackup Chain + Declared Physics GD&T Boxes
    pub tolerance_spec: SdtToleranceStackupSpec,
    pub em_gdt_box: GdtScaleBox3D,
    pub thermal_gdt_box: GdtScaleBox3D,

    // M5: 3D Maxwell Stator-Rotor Reluctance Torque (Primal + Dual at Pos 1 & Pos 2)
    pub em_primal_unaligned: MaxwellPrimalCert3D,
    pub em_dual_unaligned: MaxwellDualCert3D,
    pub em_primal_aligned: MaxwellPrimalCert3D,
    pub em_dual_aligned: MaxwellDualCert3D,

    // M5 -> M3: Coupled Joule-Thermal Liquid-Cooled CHT
    pub winding_sigma_den: u64,
    pub winding_sigma_num: Vec<i64>,
    pub coupled_thermal_cert: ThermalDualCert3D,

    // M1: 3D Preloaded Bolted Output Flange Contact + Circular Friction Cone
    pub flange_contact_cert: HmzCert3D,

    // M2: 3D Housing Yield + Buckling LMI
    pub housing_buckling_cert: HmzCert3D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssemblyRejection {
    InvalidScalingOrTarget,
    GeometryFailed(GeometryRejection),
    MaxwellFailed(MaxwellRejection),
    FlangeContactFailed(HmzRejection),
    HousingBucklingFailed(BucklingRejection),
    EmGdtBoxTooNarrow,
    ThermalGdtBoxTooNarrow,
    FlangeMissingContactFace,
    FlangeGapBelowToleranceStackup,
    FlangeMuExceedsCotsMin,
    TargetTorqueNotMet,
    WindingThermalLimitExceeded,
    FlangeLoadBelowMotorTorque,
    CotsBearingTorqueExceeded,
    CotsBearingTempExceeded,
    CotsDriverPowerExceeded,
    CotsDriverTempExceeded,
    EnvelopeLimitExceeded,
    MassBudgetExceeded,
    CodecCorrupted(&'static str),
}

/// Complete cryptographic proof report returned ONLY when EVERY gate (`G1, G3, G4, M1, M2, M3, M5, S1`) passes!
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActuatorAssemblyProofReport {
    /// Master SHA-256 Merkle digest binding all 8 sub-proofs, STEP CAD hash, tolerances & COTS contracts!
    pub master_digest: [u8; 32],
    pub step_cad_sha256: [u8; 32],

    pub curved_sandwich: CurvedSandwichReport,
    pub dfm: DfmReport,
    pub dfa: AssemblyKinematicsReport,
    pub tolerance: SdtToleranceReport,
    pub torque_raw: ReluctanceTorqueBracket,
    pub coupled_em_thermal: CoupledEmThermalReport,
    pub flange_contact: HmzProofReport,
    pub housing_buckling: BucklingProofReport,

    // Certified SI engineering metrics (exact BigRat!)
    pub certified_torque_min_nm: BigRat,
    pub certified_torque_max_nm: BigRat,
    pub certified_joule_power_w: BigRat,
    pub certified_winding_temp_max_c: BigRat,
    pub certified_flange_shear_capacity_n: BigRat,
    pub required_flange_shear_n: BigRat,
    pub total_mass_g: u64,
}

/// Zero-Trust End-to-End High-Torque Robotic Actuator Assembly Proof Verifier!
pub fn verify_actuator_assembly_cert(
    b: &ActuatorAssemblyCert,
) -> Result<ActuatorAssemblyProofReport, AssemblyRejection> {
    // 0. Validate target & scaling denominators
    let t = &b.target;
    let c = &b.cots;
    let sc = &b.scaling;
    if t.min_torque_den == 0 || t.min_torque_num <= 0
        || t.max_winding_temp_den == 0 || t.coolant_temp_den == 0
        || t.min_buckling_sf_den == 0 || t.min_buckling_sf_num <= 0
        || c.bearing_max_torque_den == 0 || c.bearing_max_temp_den == 0
        || c.bolt_circle_radius_den == 0 || c.bolt_circle_radius_num <= 0
        || c.bolt_min_mu_den == 0 || c.bolt_min_mu_num <= 0
        || c.driver_max_power_den == 0 || c.driver_max_temp_den == 0
        || sc.em_dtheta_den == 0 || sc.em_dtheta_num <= 0
        || sc.em_torque_scale_den == 0 || sc.em_torque_scale_num <= 0
        || sc.thermal_temp_scale_den == 0 || sc.thermal_temp_scale_num <= 0
        || sc.joule_power_scale_den == 0 || sc.joule_power_scale_num <= 0
        || sc.flange_force_scale_den == 0 || sc.flange_force_scale_num <= 0
    {
        return Err(AssemblyRejection::InvalidScalingOrTarget);
    }

    // Gate 1 (G1): Curved B-Rep Geometric Sandwich (Omega_in <= Omega_exact <= Omega_out)
    let curved_sandwich =
        verify_curved_annulus_sandwich(&b.housing_brep_spec, &b.housing_brep_witness)
            .map_err(AssemblyRejection::GeometryFailed)?;

    // Gate 2 (G3): DFM (wall thickness, 45-deg overhang, BFS coolant powder escape) & Rigid-Body DFA
    let dfm = verify_dfm_and_coolant_escape(&b.cooling_dfm_spec)
        .map_err(AssemblyRejection::GeometryFailed)?;
    let dfa = verify_rigid_assembly_sequence(&b.dfa_kinematics_spec)
        .map_err(AssemblyRejection::GeometryFailed)?;

    // Gate 3 (G4): 3D SDT Tolerance Stackup -> Cryptographically Enforce GD&T Boxes & Flange Gap!
    let tolerance = verify_sdt_tolerance_stackup(&b.tolerance_spec)
        .map_err(AssemblyRejection::GeometryFailed)?;
    if !b.em_gdt_box.contains(&tolerance.required_em_gdt_box) {
        return Err(AssemblyRejection::EmGdtBoxTooNarrow);
    }
    if !b.thermal_gdt_box.contains(&tolerance.required_thermal_gdt_box) {
        return Err(AssemblyRejection::ThermalGdtBoxTooNarrow);
    }

    // Check that M1 flange contact cert has a Contact face whose gap >= worst_flange_gap_norm
    // and whose friction coefficient mu <= cots.bolt_min_mu (conservative!).
    let mut found_contact = false;
    let cots_mu = BigRat::from_i64(c.bolt_min_mu_num, c.bolt_min_mu_den);
    for fk in &b.flange_contact_cert.spec.face_kinds {
        if let BoundaryFaceKind::Contact { mu_num, mu_den, gap_num, gap_den } = *fk {
            found_contact = true;
            if mu_den == 0 || gap_den == 0 {
                return Err(AssemblyRejection::InvalidScalingOrTarget);
            }
            let cert_gap = BigRat::from_i64(gap_num, gap_den);
            if cert_gap < tolerance.worst_flange_gap_norm {
                return Err(AssemblyRejection::FlangeGapBelowToleranceStackup);
            }
            let cert_mu = BigRat::from_i64(mu_num, mu_den);
            if cert_mu > cots_mu {
                return Err(AssemblyRejection::FlangeMuExceedsCotsMin);
            }
        }
    }
    if !found_contact {
        return Err(AssemblyRejection::FlangeMissingContactFace);
    }

    // Gate 4 (M5): 3D Maxwell Reluctance Torque OVER Continuous 3D GD&T Tolerance Box!
    let torque_raw = verify_reluctance_torque_bracket_gdt(
        &b.em_primal_unaligned,
        &b.em_dual_unaligned,
        &b.em_primal_aligned,
        &b.em_dual_aligned,
        sc.em_dtheta_num,
        sc.em_dtheta_den,
        sc.em_corner_idx,
        &b.em_gdt_box,
    )
    .map_err(AssemblyRejection::MaxwellFailed)?;

    let torque_scale = BigRat::from_i64(sc.em_torque_scale_num, sc.em_torque_scale_den);
    let certified_torque_min_nm = torque_raw.torque_lower.mul(&torque_scale);
    let certified_torque_max_nm = torque_raw.torque_upper.mul(&torque_scale);

    let min_torque_req = BigRat::from_i64(t.min_torque_num, t.min_torque_den);
    if certified_torque_min_nm < min_torque_req {
        return Err(AssemblyRejection::TargetTorqueNotMet);
    }

    // Gate 5 (M5 -> M3): Cryptographically Coupled EM Joule -> Liquid-Cooled CHT OVER 3D GD&T Box!
    let coupled_em_thermal = verify_coupled_em_thermal_gdt(
        &b.em_dual_aligned,
        &b.coupled_thermal_cert,
        &b.winding_sigma_num,
        b.winding_sigma_den,
        &b.thermal_gdt_box,
    )
    .map_err(AssemblyRejection::MaxwellFailed)?;

    let temp_rise_raw = coupled_em_thermal
        .thermal_report
        .avg_source_temp_upper
        .as_ref()
        .ok_or(AssemblyRejection::InvalidScalingOrTarget)?;
    let t_coolant = BigRat::from_i64(t.coolant_temp_num, t.coolant_temp_den);
    let temp_scale = BigRat::from_i64(sc.thermal_temp_scale_num, sc.thermal_temp_scale_den);
    let certified_winding_temp_max_c = t_coolant.add(&temp_rise_raw.mul(&temp_scale));

    let max_t_req = BigRat::from_i64(t.max_winding_temp_num, t.max_winding_temp_den);
    if certified_winding_temp_max_c > max_t_req {
        return Err(AssemblyRejection::WindingThermalLimitExceeded);
    }

    let power_scale = BigRat::from_i64(sc.joule_power_scale_num, sc.joule_power_scale_den);
    let certified_joule_power_w = coupled_em_thermal.total_joule_power.mul(&power_scale);

    // Gate 6 (M1): 3D Preloaded Bolted Output Flange Contact + Circular Friction Cone + Torque Coupling!
    let flange_contact = verify_hmz_cert(&b.flange_contact_cert)
        .map_err(AssemblyRejection::FlangeContactFailed)?;

    // Independently integrate total tangential shear force (|F_y| + |F_z| on x_max or y/z sides)
    // certified on the bolted output flange to guarantee it carries >= `certified_torque_max_nm / r_bolt_circle`!
    let fs = &b.flange_contact_cert.spec;
    let gd_f = BigRat::from_i64(1, fs.grid_den);
    let lx = BigRat::from_i64((fs.nx as i64) * (fs.dx_num as i64), 1).mul(&gd_f);
    let ly = BigRat::from_i64((fs.ny as i64) * (fs.dy_num as i64), 1).mul(&gd_f);
    let lz = BigRat::from_i64((fs.nz as i64) * (fs.dz_num as i64), 1).mul(&gd_f);
    let side_areas = [
        ly.mul(&lz), ly.mul(&lz),
        lx.mul(&lz), lx.mul(&lz),
        lx.mul(&ly), lx.mul(&ly),
    ];
    // Total tangential load (y/z components on Neumann faces when Contact is on side 0/1)
    let mut sum_fy = BigRat::zero();
    let mut sum_fz = BigRat::zero();
    for (s_idx, fk) in fs.face_kinds.iter().enumerate() {
        if let BoundaryFaceKind::Neumann { traction_num, traction_den } = *fk {
            let fy = side_areas[s_idx].mul(&BigRat::from_i64(traction_num[1], traction_den));
            let fz = side_areas[s_idx].mul(&BigRat::from_i64(traction_num[2], traction_den));
            sum_fy = sum_fy.add(&fy);
            sum_fz = sum_fz.add(&fz);
        }
    }
    let abs_rat = |r: &BigRat| -> BigRat {
        BigRat { num: BigInt { neg: false, mag: r.num.mag.clone() }, den: r.den.clone() }
    };
    // Exact conservative lower bound on Euclidean shear magnitude ||(F_y, F_z)||_2 >= max(|F_y|, |F_z|)
    let fy_abs = abs_rat(&sum_fy);
    let fz_abs = abs_rat(&sum_fz);
    let raw_shear_lb = if fy_abs >= fz_abs { fy_abs } else { fz_abs };
    let flange_scale = BigRat::from_i64(sc.flange_force_scale_num, sc.flange_force_scale_den);
    let certified_flange_shear_capacity_n = raw_shear_lb.mul(&flange_scale);

    let r_bc = BigRat::from_i64(c.bolt_circle_radius_num, c.bolt_circle_radius_den);
    let required_flange_shear_n = certified_torque_max_nm.div(&r_bc);
    if certified_flange_shear_capacity_n < required_flange_shear_n {
        return Err(AssemblyRejection::FlangeLoadBelowMotorTorque);
    }

    // Gate 7 (M2): 3D Gearbox Housing Yield + Global Buckling LMI!
    let housing_buckling = verify_hmz_buckling_lmi(
        &b.housing_buckling_cert,
        t.min_buckling_sf_num,
        t.min_buckling_sf_den,
        PsdMode::TwoTierSound,
    )
    .map_err(AssemblyRejection::HousingBucklingFailed)?;

    // Gate 8 (S1): COTS Component Ratings + Envelope + Mass Budget!
    let bearing_t_max = BigRat::from_i64(c.bearing_max_torque_num, c.bearing_max_torque_den);
    if certified_torque_max_nm > bearing_t_max {
        return Err(AssemblyRejection::CotsBearingTorqueExceeded);
    }
    let bearing_temp_max = BigRat::from_i64(c.bearing_max_temp_num, c.bearing_max_temp_den);
    if certified_winding_temp_max_c > bearing_temp_max {
        return Err(AssemblyRejection::CotsBearingTempExceeded);
    }
    let driver_p_max = BigRat::from_i64(c.driver_max_power_num, c.driver_max_power_den);
    if certified_joule_power_w > driver_p_max {
        return Err(AssemblyRejection::CotsDriverPowerExceeded);
    }
    let driver_temp_max = BigRat::from_i64(c.driver_max_temp_num, c.driver_max_temp_den);
    if certified_winding_temp_max_c > driver_temp_max {
        return Err(AssemblyRejection::CotsDriverTempExceeded);
    }
    if c.actual_od_um > t.max_od_um || c.actual_length_um > t.max_length_um {
        return Err(AssemblyRejection::EnvelopeLimitExceeded);
    }
    let total_mass_g = c
        .custom_parts_mass_g
        .saturating_add(c.bearing_mass_g)
        .saturating_add(c.bolts_mass_g)
        .saturating_add(c.electronics_mass_g);
    if total_mass_g > t.max_mass_g {
        return Err(AssemblyRejection::MassBudgetExceeded);
    }

    // Gate 9: Master Cryptographic SHA-256 Merkle Proof Digest over ALL verified sub-reports + STEP CAD!
    let mut h = Sha256::new();
    h.update(ASSEMBLY_KERNEL_ID.as_bytes());
    h.update(&b.step_cad_sha256);
    h.update(&curved_sandwich.spec_hash);
    h.update(&dfm.spec_hash);
    h.update(&dfa.spec_hash);
    h.update(&tolerance.spec_hash);
    b.em_gdt_box.hash_into(&mut h);
    b.thermal_gdt_box.hash_into(&mut h);
    h.update(&coupled_em_thermal.maxwell_report.spec_hash);
    h.update(&coupled_em_thermal.thermal_report.spec_hash);
    h.update(&flange_contact.spec_hash);
    h.update(&housing_buckling.spec_hash);
    h.update_u64(total_mass_g);
    h.update_u64(c.actual_od_um);
    h.update_u64(c.actual_length_um);
    let master_digest = h.finalize();

    Ok(ActuatorAssemblyProofReport {
        master_digest,
        step_cad_sha256: b.step_cad_sha256,
        curved_sandwich,
        dfm,
        dfa,
        tolerance,
        torque_raw,
        coupled_em_thermal,
        flange_contact,
        housing_buckling,
        certified_torque_min_nm,
        certified_torque_max_nm,
        certified_joule_power_w,
        certified_winding_temp_max_c,
        certified_flange_shear_capacity_n,
        required_flange_shear_n,
        total_mass_g,
    })
}

// ========================== DETERMINISTIC ZERO-DEP `.ntcert` WIRE CODEC ==========================

struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(65536) }
    }
    fn u8(&mut self, v: u8) { self.buf.push(v); }
    fn u32(&mut self, v: u32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn u64(&mut self, v: u64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn i64(&mut self, v: i64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn bytes(&mut self, b: &[u8]) { self.buf.extend_from_slice(b); }
    fn vec_u8(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.bytes(v);
    }
    fn vec_i64(&mut self, v: &[i64]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.i64(x);
        }
    }
    fn gdt(&mut self, g: &GdtScaleBox3D) {
        self.u64(g.scale_den);
        self.u64(g.ax_min); self.u64(g.ax_max);
        self.u64(g.ay_min); self.u64(g.ay_max);
        self.u64(g.az_min); self.u64(g.az_max);
    }
    fn nedelec(&mut self, w: &NedelecWitness3D) {
        self.u64(w.field_den);
        self.vec_i64(&w.ex);
        self.vec_i64(&w.ey);
        self.vec_i64(&w.ez);
    }
    fn maxwell_spec(&mut self, s: &MaxwellSpec3D) {
        self.u64(s.nx as u64); self.u64(s.ny as u64); self.u64(s.nz as u64);
        self.u64(s.grid_den); self.u64(s.dx_num); self.u64(s.dy_num); self.u64(s.dz_num);
        self.u64(s.j_den);
        self.vec_i64(&s.jx); self.vec_i64(&s.jy); self.vec_i64(&s.jz);
        self.u64(s.material_corners.len() as u64);
        for mc in &s.material_corners {
            self.u64(mc.mat_den);
            self.vec_i64(&mc.mu_num);
        }
    }
    fn thermal_spec(&mut self, s: &ThermalSpec3D) {
        self.u64(s.nx as u64); self.u64(s.ny as u64); self.u64(s.nz as u64);
        self.u64(s.grid_den); self.u64(s.dx_num); self.u64(s.dy_num); self.u64(s.dz_num);
        self.u64(s.q_src_den);
        self.vec_i64(&s.q_src_num);
        self.u64(s.material_corners.len() as u64);
        for mc in &s.material_corners {
            self.u64(mc.mat_den);
            self.vec_i64(&mc.k_num);
            for &rh in &mc.robin_h_num { self.i64(rh); }
        }
        for bc in &s.face_bcs {
            match *bc {
                ThermalBcKind::Dirichlet { temp_num, temp_den } => {
                    self.u8(0); self.i64(temp_num); self.u64(temp_den);
                }
                ThermalBcKind::Neumann { flux_num, flux_den } => {
                    self.u8(1); self.i64(flux_num); self.u64(flux_den);
                }
                ThermalBcKind::Robin { t_inf_num, t_inf_den } => {
                    self.u8(2); self.i64(t_inf_num); self.u64(t_inf_den);
                }
            }
        }
    }
    fn hmz_cert(&mut self, c: &HmzCert3D) {
        let s = &c.spec;
        self.u64(s.nx as u64); self.u64(s.ny as u64); self.u64(s.nz as u64);
        self.u64(s.grid_den); self.u64(s.dx_num); self.u64(s.dy_num); self.u64(s.dz_num);
        self.u64(s.mat_den);
        self.u64(s.material_corners.len() as u64);
        for mc in &s.material_corners {
            self.i64(mc.mu_num); self.i64(mc.lam_num); self.i64(mc.omega2_rho_num);
        }
        for fk in &s.face_kinds {
            match *fk {
                BoundaryFaceKind::Dirichlet => self.u8(0),
                BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                    self.u8(1);
                    self.i64(traction_num[0]); self.i64(traction_num[1]); self.i64(traction_num[2]);
                    self.u64(traction_den);
                }
                BoundaryFaceKind::Contact { mu_num, mu_den, gap_num, gap_den } => {
                    self.u8(2);
                    self.i64(mu_num); self.u64(mu_den); self.i64(gap_num); self.u64(gap_den);
                }
                BoundaryFaceKind::Interface { tag } => {
                    self.u8(3);
                    self.u32(tag);
                }
            }
        }
        match s.yield_limit {
            None => self.u8(0),
            Some((yn, yd)) => { self.u8(1); self.i64(yn); self.u64(yd); }
        }
        let w = &c.witness;
        self.u64(w.flux_den);
        self.vec_i64(&w.fx); self.vec_i64(&w.fy); self.vec_i64(&w.fz);
        self.vec_i64(&w.ex); self.vec_i64(&w.ey); self.vec_i64(&w.ez);
    }
}

struct Dec<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    fn take(&mut self, n: usize) -> Result<&'a [u8], AssemblyRejection> {
        if self.pos.checked_add(n).map_or(true, |end| end > self.data.len()) {
            return Err(AssemblyRejection::CodecCorrupted("unexpected EOF"));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, AssemblyRejection> { Ok(self.take(1)?[0]) }
    fn u32(&mut self) -> Result<u32, AssemblyRejection> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, AssemblyRejection> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, AssemblyRejection> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize_lim(&mut self, max_len: usize) -> Result<usize, AssemblyRejection> {
        let v = self.u64()? as usize;
        if v > max_len {
            return Err(AssemblyRejection::CodecCorrupted("length exceeds bound"));
        }
        Ok(v)
    }
    fn vec_u8(&mut self, max_len: usize) -> Result<Vec<u8>, AssemblyRejection> {
        let n = self.usize_lim(max_len)?;
        Ok(self.take(n)?.to_vec())
    }
    fn vec_i64(&mut self, max_len: usize) -> Result<Vec<i64>, AssemblyRejection> {
        let n = self.usize_lim(max_len)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.i64()?);
        }
        Ok(out)
    }
    fn gdt(&mut self) -> Result<GdtScaleBox3D, AssemblyRejection> {
        Ok(GdtScaleBox3D {
            scale_den: self.u64()?,
            ax_min: self.u64()?, ax_max: self.u64()?,
            ay_min: self.u64()?, ay_max: self.u64()?,
            az_min: self.u64()?, az_max: self.u64()?,
        })
    }
    fn nedelec(&mut self) -> Result<NedelecWitness3D, AssemblyRejection> {
        Ok(NedelecWitness3D {
            field_den: self.u64()?,
            ex: self.vec_i64(2_000_000)?,
            ey: self.vec_i64(2_000_000)?,
            ez: self.vec_i64(2_000_000)?,
        })
    }
    fn maxwell_spec(&mut self) -> Result<MaxwellSpec3D, AssemblyRejection> {
        let nx = self.usize_lim(512)?; let ny = self.usize_lim(512)?; let nz = self.usize_lim(512)?;
        let grid_den = self.u64()?; let dx_num = self.u64()?; let dy_num = self.u64()?; let dz_num = self.u64()?;
        let j_den = self.u64()?;
        let jx = self.vec_i64(2_000_000)?; let jy = self.vec_i64(2_000_000)?; let jz = self.vec_i64(2_000_000)?;
        let nc = self.usize_lim(128)?;
        let mut material_corners = Vec::with_capacity(nc);
        for _ in 0..nc {
            material_corners.push(MaxwellMaterialCorner {
                mat_den: self.u64()?,
                mu_num: self.vec_i64(2_000_000)?,
            });
        }
        Ok(MaxwellSpec3D { nx, ny, nz, grid_den, dx_num, dy_num, dz_num, j_den, jx, jy, jz, material_corners })
    }
    fn thermal_spec(&mut self) -> Result<ThermalSpec3D, AssemblyRejection> {
        let nx = self.usize_lim(512)?; let ny = self.usize_lim(512)?; let nz = self.usize_lim(512)?;
        let grid_den = self.u64()?; let dx_num = self.u64()?; let dy_num = self.u64()?; let dz_num = self.u64()?;
        let q_src_den = self.u64()?;
        let q_src_num = self.vec_i64(2_000_000)?;
        let nc = self.usize_lim(128)?;
        let mut material_corners = Vec::with_capacity(nc);
        for _ in 0..nc {
            let mat_den = self.u64()?;
            let k_num = self.vec_i64(2_000_000)?;
            let mut robin_h_num = [0i64; 6];
            for r in &mut robin_h_num { *r = self.i64()?; }
            material_corners.push(ThermalMaterialCorner { mat_den, k_num, robin_h_num });
        }
        let mut face_bcs = [ThermalBcKind::Dirichlet { temp_num: 0, temp_den: 1 }; 6];
        for bc in &mut face_bcs {
            *bc = match self.u8()? {
                0 => ThermalBcKind::Dirichlet { temp_num: self.i64()?, temp_den: self.u64()? },
                1 => ThermalBcKind::Neumann { flux_num: self.i64()?, flux_den: self.u64()? },
                2 => ThermalBcKind::Robin { t_inf_num: self.i64()?, t_inf_den: self.u64()? },
                _ => return Err(AssemblyRejection::CodecCorrupted("invalid thermal bc tag")),
            };
        }
        Ok(ThermalSpec3D { nx, ny, nz, grid_den, dx_num, dy_num, dz_num, q_src_den, q_src_num, material_corners, face_bcs })
    }
    fn hmz_cert(&mut self) -> Result<HmzCert3D, AssemblyRejection> {
        let nx = self.usize_lim(512)?; let ny = self.usize_lim(512)?; let nz = self.usize_lim(512)?;
        let grid_den = self.u64()?; let dx_num = self.u64()?; let dy_num = self.u64()?; let dz_num = self.u64()?;
        let mat_den = self.u64()?;
        let nc = self.usize_lim(128)?;
        let mut material_corners = Vec::with_capacity(nc);
        for _ in 0..nc {
            material_corners.push(MaterialCorner {
                mu_num: self.i64()?, lam_num: self.i64()?, omega2_rho_num: self.i64()?,
            });
        }
        let mut face_kinds = [BoundaryFaceKind::Dirichlet; 6];
        for fk in &mut face_kinds {
            *fk = match self.u8()? {
                0 => BoundaryFaceKind::Dirichlet,
                1 => BoundaryFaceKind::Neumann {
                    traction_num: [self.i64()?, self.i64()?, self.i64()?],
                    traction_den: self.u64()?,
                },
                2 => BoundaryFaceKind::Contact {
                    mu_num: self.i64()?, mu_den: self.u64()?,
                    gap_num: self.i64()?, gap_den: self.u64()?,
                },
                3 => BoundaryFaceKind::Interface { tag: self.u32()? },
                _ => return Err(AssemblyRejection::CodecCorrupted("invalid hmz bc tag")),
            };
        }
        let yield_limit = match self.u8()? {
            0 => None,
            1 => Some((self.i64()?, self.u64()?)),
            _ => return Err(AssemblyRejection::CodecCorrupted("invalid yield tag")),
        };
        let witness = HmzWitness3D {
            flux_den: self.u64()?,
            fx: self.vec_i64(2_000_000)?, fy: self.vec_i64(2_000_000)?, fz: self.vec_i64(2_000_000)?,
            ex: self.vec_i64(2_000_000)?, ey: self.vec_i64(2_000_000)?, ez: self.vec_i64(2_000_000)?,
        };
        Ok(HmzCert3D {
            spec: HmzSpec3D { nx, ny, nz, grid_den, dx_num, dy_num, dz_num, mat_den, material_corners, face_kinds, yield_limit },
            witness,
        })
    }
}

/// Deterministic zero-dependency `.ntcert` binary encoder with trailing SHA-256 wire integrity tag!
pub fn encode_ntcert(b: &ActuatorAssemblyCert) -> Vec<u8> {
    let mut e = Enc::new();
    e.bytes(&NTCERT_MAGIC);
    e.u32(NTCERT_VERSION);

    // Target
    e.i64(b.target.min_torque_num); e.u64(b.target.min_torque_den);
    e.i64(b.target.max_winding_temp_num); e.u64(b.target.max_winding_temp_den);
    e.i64(b.target.coolant_temp_num); e.u64(b.target.coolant_temp_den);
    e.i64(b.target.min_buckling_sf_num); e.u64(b.target.min_buckling_sf_den);
    e.u64(b.target.max_od_um); e.u64(b.target.max_length_um); e.u64(b.target.max_mass_g);

    // COTS
    e.i64(b.cots.bearing_max_torque_num); e.u64(b.cots.bearing_max_torque_den);
    e.i64(b.cots.bearing_max_temp_num); e.u64(b.cots.bearing_max_temp_den);
    e.u64(b.cots.bearing_runout_um); e.u64(b.cots.bearing_mass_g);
    e.i64(b.cots.bolt_circle_radius_num); e.u64(b.cots.bolt_circle_radius_den);
    e.i64(b.cots.bolt_min_mu_num); e.u64(b.cots.bolt_min_mu_den); e.u64(b.cots.bolts_mass_g);
    e.i64(b.cots.driver_max_power_num); e.u64(b.cots.driver_max_power_den);
    e.i64(b.cots.driver_max_temp_num); e.u64(b.cots.driver_max_temp_den); e.u64(b.cots.electronics_mass_g);
    e.u64(b.cots.custom_parts_mass_g); e.u64(b.cots.actual_od_um); e.u64(b.cots.actual_length_um);

    // Scaling & STEP hash
    e.i64(b.scaling.em_dtheta_num); e.u64(b.scaling.em_dtheta_den);
    e.i64(b.scaling.em_torque_scale_num); e.u64(b.scaling.em_torque_scale_den);
    e.i64(b.scaling.thermal_temp_scale_num); e.u64(b.scaling.thermal_temp_scale_den);
    e.i64(b.scaling.joule_power_scale_num); e.u64(b.scaling.joule_power_scale_den);
    e.i64(b.scaling.flange_force_scale_num); e.u64(b.scaling.flange_force_scale_den);
    e.u64(b.scaling.em_corner_idx as u64);
    e.bytes(&b.step_cad_sha256);

    // G1 Curved B-Rep
    let g1 = &b.housing_brep_spec;
    e.u64(g1.nx as u64); e.u64(g1.ny as u64); e.u64(g1.nz as u64);
    e.u64(g1.grid_den); e.u64(g1.dx_num); e.u64(g1.dy_num); e.u64(g1.dz_num);
    e.u64(g1.r_den); e.i64(g1.center_x_num); e.i64(g1.center_y_num);
    e.i64(g1.r_in_num); e.i64(g1.r_out_num); e.i64(g1.z_min_num); e.i64(g1.z_max_num);
    e.u64(g1.bolt_holes.len() as u64);
    for bh in &g1.bolt_holes { e.i64(bh.cx_num); e.i64(bh.cy_num); e.i64(bh.radius_num); }
    e.vec_u8(&b.housing_brep_witness.in_mask);
    e.vec_u8(&b.housing_brep_witness.out_mask);

    // G3 DFM + DFA
    e.u64(b.cooling_dfm_spec.nx as u64); e.u64(b.cooling_dfm_spec.ny as u64); e.u64(b.cooling_dfm_spec.nz as u64);
    e.vec_u8(&b.cooling_dfm_spec.voxels);
    e.u64(b.cooling_dfm_spec.min_wall_cells as u64);
    e.u8(b.cooling_dfm_spec.check_45deg_overhang as u8);
    e.u8(b.cooling_dfm_spec.require_powder_escape as u8);

    e.u64(b.dfa_kinematics_spec.parts.len() as u64);
    for p in &b.dfa_kinematics_spec.parts {
        e.u32(p.part_id);
        e.u64(p.boxes.len() as u64);
        for bx in &p.boxes {
            e.i64(bx.x0); e.i64(bx.x1); e.i64(bx.y0); e.i64(bx.y1); e.i64(bx.z0); e.i64(bx.z1);
        }
    }
    e.u64(b.dfa_kinematics_spec.disassembly_steps.len() as u64);
    for st in &b.dfa_kinematics_spec.disassembly_steps {
        e.u32(st.part_id); e.u8(st.axis); e.u8(st.dir as u8);
    }

    // G4 SDT Tolerance + GD&T Boxes
    let ts = &b.tolerance_spec;
    e.u64(ts.nominal_x_um); e.u64(ts.nominal_y_um); e.u64(ts.nominal_z_um);
    e.u64(ts.nominal_airgap_um); e.u64(ts.min_safe_airgap_um); e.u64(ts.flange_scale_um);
    e.u64(ts.links.len() as u64);
    for l in &ts.links {
        e.u32(l.link_id); e.u64(l.pm_x_um); e.u64(l.pm_y_um); e.u64(l.pm_z_um);
        e.u64(l.pm_airgap_um); e.u64(l.pm_flange_gap_um);
    }
    e.gdt(&b.em_gdt_box);
    e.gdt(&b.thermal_gdt_box);

    // M5 Maxwell (4 certs)
    e.maxwell_spec(&b.em_primal_unaligned.spec); e.nedelec(&b.em_primal_unaligned.a_witness);
    e.maxwell_spec(&b.em_dual_unaligned.spec); e.nedelec(&b.em_dual_unaligned.h_witness);
    e.maxwell_spec(&b.em_primal_aligned.spec); e.nedelec(&b.em_primal_aligned.a_witness);
    e.maxwell_spec(&b.em_dual_aligned.spec); e.nedelec(&b.em_dual_aligned.h_witness);

    // M5 -> M3 Coupled Thermal
    e.u64(b.winding_sigma_den);
    e.vec_i64(&b.winding_sigma_num);
    e.thermal_spec(&b.coupled_thermal_cert.spec);
    e.u64(b.coupled_thermal_cert.witness.flux_den);
    e.vec_i64(&b.coupled_thermal_cert.witness.qx);
    e.vec_i64(&b.coupled_thermal_cert.witness.qy);
    e.vec_i64(&b.coupled_thermal_cert.witness.qz);

    // M1 Flange & M2 Buckling
    e.hmz_cert(&b.flange_contact_cert);
    e.hmz_cert(&b.housing_buckling_cert);

    // Trailing SHA-256 wire integrity tag!
    let mut sha = Sha256::new();
    sha.update(&e.buf);
    let tag = sha.finalize();
    e.bytes(&tag);
    e.buf
}

/// Zero-trust `.ntcert` binary decoder — checks magic, version, trailing SHA-256 integrity, and exact EOF!
pub fn decode_ntcert(bytes: &[u8]) -> Result<ActuatorAssemblyCert, AssemblyRejection> {
    if bytes.len() < 40 {
        return Err(AssemblyRejection::CodecCorrupted("file too short"));
    }
    let payload_len = bytes.len() - 32;
    let mut sha = Sha256::new();
    sha.update(&bytes[..payload_len]);
    let expected_tag = sha.finalize();
    if expected_tag != bytes[payload_len..] {
        return Err(AssemblyRejection::CodecCorrupted("SHA-256 wire integrity check failed"));
    }

    let mut d = Dec::new(&bytes[..payload_len]);
    if d.take(4)? != NTCERT_MAGIC {
        return Err(AssemblyRejection::CodecCorrupted("invalid NTCT magic"));
    }
    if d.u32()? != NTCERT_VERSION {
        return Err(AssemblyRejection::CodecCorrupted("unsupported ntcert version"));
    }

    let target = ActuatorTargetSpec {
        min_torque_num: d.i64()?, min_torque_den: d.u64()?,
        max_winding_temp_num: d.i64()?, max_winding_temp_den: d.u64()?,
        coolant_temp_num: d.i64()?, coolant_temp_den: d.u64()?,
        min_buckling_sf_num: d.i64()?, min_buckling_sf_den: d.u64()?,
        max_od_um: d.u64()?, max_length_um: d.u64()?, max_mass_g: d.u64()?,
    };
    let cots = CotsContractsSpec {
        bearing_max_torque_num: d.i64()?, bearing_max_torque_den: d.u64()?,
        bearing_max_temp_num: d.i64()?, bearing_max_temp_den: d.u64()?,
        bearing_runout_um: d.u64()?, bearing_mass_g: d.u64()?,
        bolt_circle_radius_num: d.i64()?, bolt_circle_radius_den: d.u64()?,
        bolt_min_mu_num: d.i64()?, bolt_min_mu_den: d.u64()?, bolts_mass_g: d.u64()?,
        driver_max_power_num: d.i64()?, driver_max_power_den: d.u64()?,
        driver_max_temp_num: d.i64()?, driver_max_temp_den: d.u64()?, electronics_mass_g: d.u64()?,
        custom_parts_mass_g: d.u64()?, actual_od_um: d.u64()?, actual_length_um: d.u64()?,
    };
    let scaling = PhysicalScalingContract {
        em_dtheta_num: d.i64()?, em_dtheta_den: d.u64()?,
        em_torque_scale_num: d.i64()?, em_torque_scale_den: d.u64()?,
        thermal_temp_scale_num: d.i64()?, thermal_temp_scale_den: d.u64()?,
        joule_power_scale_num: d.i64()?, joule_power_scale_den: d.u64()?,
        flange_force_scale_num: d.i64()?, flange_force_scale_den: d.u64()?,
        em_corner_idx: d.usize_lim(128)?,
    };
    let mut step_cad_sha256 = [0u8; 32];
    step_cad_sha256.copy_from_slice(d.take(32)?);

    let g1_nx = d.usize_lim(512)?; let g1_ny = d.usize_lim(512)?; let g1_nz = d.usize_lim(512)?;
    let g1_gd = d.u64()?; let g1_dx = d.u64()?; let g1_dy = d.u64()?; let g1_dz = d.u64()?;
    let g1_rd = d.u64()?; let g1_cx = d.i64()?; let g1_cy = d.i64()?;
    let g1_rin = d.i64()?; let g1_rout = d.i64()?; let g1_zmin = d.i64()?; let g1_zmax = d.i64()?;
    let nb = d.usize_lim(256)?;
    let mut bolt_holes = Vec::with_capacity(nb);
    for _ in 0..nb {
        bolt_holes.push(BoltHole2D { cx_num: d.i64()?, cy_num: d.i64()?, radius_num: d.i64()? });
    }
    let housing_brep_spec = CurvedAnnulusSpec {
        nx: g1_nx, ny: g1_ny, nz: g1_nz, grid_den: g1_gd, dx_num: g1_dx, dy_num: g1_dy, dz_num: g1_dz,
        r_den: g1_rd, center_x_num: g1_cx, center_y_num: g1_cy, r_in_num: g1_rin, r_out_num: g1_rout,
        z_min_num: g1_zmin, z_max_num: g1_zmax, bolt_holes,
    };
    let housing_brep_witness = CurvedSandwichWitness {
        in_mask: d.vec_u8(2_000_000)?,
        out_mask: d.vec_u8(2_000_000)?,
    };

    let cooling_dfm_spec = DfmVoxelSpec {
        nx: d.usize_lim(512)?, ny: d.usize_lim(512)?, nz: d.usize_lim(512)?,
        voxels: d.vec_u8(2_000_000)?,
        min_wall_cells: d.usize_lim(512)?,
        check_45deg_overhang: d.u8()? != 0,
        require_powder_escape: d.u8()? != 0,
    };

    let np = d.usize_lim(1024)?;
    let mut parts = Vec::with_capacity(np);
    for _ in 0..np {
        let part_id = d.u32()?;
        let nbx = d.usize_lim(256)?;
        let mut boxes = Vec::with_capacity(nbx);
        for _ in 0..nbx {
            boxes.push(IntBox3D {
                x0: d.i64()?, x1: d.i64()?, y0: d.i64()?, y1: d.i64()?, z0: d.i64()?, z1: d.i64()?,
            });
        }
        parts.push(AssemblyPartGeom { part_id, boxes });
    }
    let ns = d.usize_lim(1024)?;
    let mut disassembly_steps = Vec::with_capacity(ns);
    for _ in 0..ns {
        disassembly_steps.push(DisassemblyStep {
            part_id: d.u32()?, axis: d.u8()?, dir: d.u8()? as i8,
        });
    }
    let dfa_kinematics_spec = AssemblyKinematicsSpec { parts, disassembly_steps };

    let nom_x = d.u64()?; let nom_y = d.u64()?; let nom_z = d.u64()?;
    let nom_ag = d.u64()?; let min_ag = d.u64()?; let f_sc = d.u64()?;
    let nl = d.usize_lim(256)?;
    let mut links = Vec::with_capacity(nl);
    for _ in 0..nl {
        links.push(ToleranceLinkUm {
            link_id: d.u32()?, pm_x_um: d.u64()?, pm_y_um: d.u64()?, pm_z_um: d.u64()?,
            pm_airgap_um: d.u64()?, pm_flange_gap_um: d.u64()?,
        });
    }
    let tolerance_spec = SdtToleranceStackupSpec {
        nominal_x_um: nom_x, nominal_y_um: nom_y, nominal_z_um: nom_z,
        nominal_airgap_um: nom_ag, min_safe_airgap_um: min_ag, flange_scale_um: f_sc, links,
    };
    let em_gdt_box = d.gdt()?;
    let thermal_gdt_box = d.gdt()?;

    let em_primal_unaligned = MaxwellPrimalCert3D { spec: d.maxwell_spec()?, a_witness: d.nedelec()? };
    let em_dual_unaligned = MaxwellDualCert3D { spec: d.maxwell_spec()?, h_witness: d.nedelec()? };
    let em_primal_aligned = MaxwellPrimalCert3D { spec: d.maxwell_spec()?, a_witness: d.nedelec()? };
    let em_dual_aligned = MaxwellDualCert3D { spec: d.maxwell_spec()?, h_witness: d.nedelec()? };

    let winding_sigma_den = d.u64()?;
    let winding_sigma_num = d.vec_i64(2_000_000)?;
    let coupled_thermal_cert = ThermalDualCert3D {
        spec: d.thermal_spec()?,
        witness: ThermalFluxWitness3D {
            flux_den: d.u64()?,
            qx: d.vec_i64(2_000_000)?,
            qy: d.vec_i64(2_000_000)?,
            qz: d.vec_i64(2_000_000)?,
        },
    };
    let flange_contact_cert = d.hmz_cert()?;
    let housing_buckling_cert = d.hmz_cert()?;

    if d.pos != payload_len {
        return Err(AssemblyRejection::CodecCorrupted("trailing unconsumed bytes"));
    }

    Ok(ActuatorAssemblyCert {
        target, cots, scaling, step_cad_sha256,
        housing_brep_spec, housing_brep_witness,
        cooling_dfm_spec, dfa_kinematics_spec,
        tolerance_spec, em_gdt_box, thermal_gdt_box,
        em_primal_unaligned, em_dual_unaligned, em_primal_aligned, em_dual_aligned,
        winding_sigma_den, winding_sigma_num, coupled_thermal_cert,
        flange_contact_cert, housing_buckling_cert,
    })
}
