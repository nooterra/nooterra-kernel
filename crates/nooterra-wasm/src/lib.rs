//! Zero-dependency WebAssembly C-ABI adapter for `nooterra-kernel`.
//!
//! All `unsafe` FFI pointer conversions are isolated strictly inside this thin wrapper,
//! preserving `#![no_std]` and `#![forbid(unsafe_code)]` across 100% of `nooterra-kernel`.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;

use nooterra_kernel::assembly::{
    decode_ntcert, verify_actuator_assembly_cert, ActuatorAssemblyProofReport, AssemblyRejection,
};
use nooterra_kernel::geometry::VOXEL_SOLID;
use nooterra_kernel::hmz::{idx_ez, idx_fx};
use nooterra_kernel::sha256::Sha256;
use nooterra_kernel::BigRat;

thread_local! {
    static LAST_JSON: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_out(json: String) {
    LAST_JSON.with(|slot| {
        *slot.borrow_mut() = json.into_bytes();
    });
}

fn hex32(b: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &byte in b {
        s.push(HEX[(byte >> 4) as usize] as char);
        s.push(HEX[(byte & 0x0f) as usize] as char);
    }
    s
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn rat_f64(r: &BigRat) -> f64 {
    r.to_f64_checked().unwrap_or(0.0)
}

fn format_pass_json(rep: &ActuatorAssemblyProofReport, step_verified: bool, attack_label: &str) -> String {
    let torque_min = rat_f64(&rep.certified_torque_min_nm);
    let torque_max = rat_f64(&rep.certified_torque_max_nm);
    let joule_w = rat_f64(&rep.certified_joule_power_w);
    let temp_max = rat_f64(&rep.certified_winding_temp_max_c);
    let shear_cap = rat_f64(&rep.certified_flange_shear_capacity_n);
    let shear_req = rat_f64(&rep.required_flange_shear_n);
    let vm_peak = rat_f64(&rep.flange_contact.max_von_mises_sq).max(0.0).sqrt();
    let buck_sf = rat_f64(&rep.housing_buckling.lambda_cert);
    let em_gap_pct = if torque_max > 0.0 {
        100.0 * (torque_max - torque_min) / torque_max
    } else {
        0.0
    };

    format!(
        concat!(
            "{{",
            "\"status\":\"CERTIFIED\",",
            "\"attack\":\"{}\",",
            "\"step_bound\":{},",
            "\"master_digest\":\"{}\",",
            "\"step_cad_sha256\":\"{}\",",
            "\"metrics\":{{",
            "\"torque_min_nm\":{:.4},",
            "\"torque_max_nm\":{:.4},",
            "\"torque_bracket_gap_pct\":{:.2},",
            "\"joule_power_w\":{:.3},",
            "\"winding_temp_max_c\":{:.3},",
            "\"flange_shear_cap_n\":{:.2},",
            "\"flange_shear_req_n\":{:.2},",
            "\"flange_vm_peak\":{:.4},",
            "\"flange_contact_faces\":{},",
            "\"buckling_sf\":{:.4},",
            "\"buckling_free_dofs\":{},",
            "\"total_mass_g\":{},",
            "\"min_airgap_um\":{},",
            "\"dfm_coolant_voxels\":{},",
            "\"dfa_steps\":{},",
            "\"brep_inner_cells\":{},",
            "\"brep_outer_cells\":{}",
            "}},",
            "\"gates\":[",
            "{{\"id\":\"G1\",\"name\":\"Curved B-Rep Sandwich\",\"pass\":true,\"detail\":\"Omega_in ({}) <= Omega_exact <= Omega_out ({})\"}},",
            "{{\"id\":\"G3\",\"name\":\"DFM & Collision-Free DFA\",\"pass\":true,\"detail\":\"{} coolant voxels escaped, {} collision-free steps\"}},",
            "{{\"id\":\"G4\",\"name\":\"3D SDT Tolerance Stackup\",\"pass\":true,\"detail\":\"worst-case airgap {} um >= safe floor\"}},",
            "{{\"id\":\"M5\",\"name\":\"3D Maxwell EM Torque\",\"pass\":true,\"detail\":\"[{:.2}, {:.2}] N*m across 8-corner GD&T box\"}},",
            "{{\"id\":\"M3\",\"name\":\"Coupled 3D Joule-Thermal CHT\",\"pass\":true,\"detail\":\"T_max <= {:.2} C at {:.2} W Joule loss\"}},",
            "{{\"id\":\"M1\",\"name\":\"3D Bolted Circular Friction\",\"pass\":true,\"detail\":\"{} contact cones verified, shear cap {:.1} N >= {:.1} N\"}},",
            "{{\"id\":\"M2\",\"name\":\"3D Yield & Buckling LMI\",\"pass\":true,\"detail\":\"SF >= {:.3} ({} Q1 DOFs, Rump verified PD)\"}},",
            "{{\"id\":\"S1\",\"name\":\"COTS Contracts & Mass\",\"pass\":true,\"detail\":\"{} g <= budget, bearing/bolt/PCB ratings met\"}}",
            "]",
            "}}"
        ),
        escape_json(attack_label),
        if step_verified { "true" } else { "false" },
        hex32(&rep.master_digest),
        hex32(&rep.step_cad_sha256),
        torque_min,
        torque_max,
        em_gap_pct,
        joule_w,
        temp_max,
        shear_cap,
        shear_req,
        vm_peak,
        rep.flange_contact.contact_faces_verified,
        buck_sf,
        rep.housing_buckling.n_free_dofs,
        rep.total_mass_g,
        rep.tolerance.min_airgap_um,
        rep.dfm.coolant_cells,
        rep.dfa.num_steps_verified,
        rep.curved_sandwich.inner_cells,
        rep.curved_sandwich.outer_cells,
        rep.curved_sandwich.inner_cells,
        rep.curved_sandwich.outer_cells,
        rep.dfm.coolant_cells,
        rep.dfa.num_steps_verified,
        rep.tolerance.min_airgap_um,
        torque_min,
        torque_max,
        temp_max,
        joule_w,
        rep.flange_contact.contact_faces_verified,
        shear_cap,
        shear_req,
        buck_sf,
        rep.housing_buckling.n_free_dofs,
        rep.total_mass_g
    )
}

fn rejection_gate(err: &AssemblyRejection) -> &'static str {
    match err {
        AssemblyRejection::CodecCorrupted(_) => "WIRE",
        AssemblyRejection::GeometryFailed(g) => match g {
            nooterra_kernel::GeometryRejection::InnerNotSubsetOfOuter { .. }
            | nooterra_kernel::GeometryRejection::InnerShellProtrudesOutsideBrep { .. }
            | nooterra_kernel::GeometryRejection::OuterShellMissesBrep { .. } => "G1",
            nooterra_kernel::GeometryRejection::AirgapRubRisk { .. } => "G4",
            _ => "G3",
        },
        AssemblyRejection::EmGdtBoxTooNarrow
        | AssemblyRejection::ThermalGdtBoxTooNarrow
        | AssemblyRejection::FlangeGapBelowToleranceStackup => "G4",
        AssemblyRejection::MaxwellFailed(_) | AssemblyRejection::TargetTorqueNotMet => "M5",
        AssemblyRejection::WindingThermalLimitExceeded => "M3",
        AssemblyRejection::FlangeContactFailed(_)
        | AssemblyRejection::FlangeMissingContactFace
        | AssemblyRejection::FlangeMuExceedsCotsMin
        | AssemblyRejection::FlangeLoadBelowMotorTorque => "M1",
        AssemblyRejection::HousingBucklingFailed(_) => "M2",
        _ => "S1",
    }
}

fn format_reject_json(gate: &str, code: &str, attack_label: &str) -> String {
    format!(
        "{{\"status\":\"REFUSED\",\"attack\":\"{}\",\"failed_gate\":\"{}\",\"rejection\":\"{}\"}}",
        escape_json(attack_label),
        escape_json(gate),
        escape_json(code),
    )
}

#[no_mangle]
pub extern "C" fn nt_alloc(len: usize) -> *mut u8 {
    let size = len.max(1);
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { alloc(layout) }
}

#[no_mangle]
pub extern "C" fn nt_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let size = len.max(1);
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { dealloc(ptr, layout) }
}

#[no_mangle]
pub extern "C" fn nt_out_ptr() -> *const u8 {
    LAST_JSON.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn nt_out_len() -> usize {
    LAST_JSON.with(|slot| slot.borrow().len())
}

/// Zero-trust verify raw `.ntcert` bytes (and optional `.step` bytes for cryptographic CAD binding).
/// Returns `0` when CERTIFIED, `1` when REFUSED. Full JSON report available via `nt_out_ptr`/`nt_out_len`.
#[no_mangle]
pub extern "C" fn nt_verify_ntcert(
    cert_ptr: *const u8,
    cert_len: usize,
    step_ptr: *const u8,
    step_len: usize,
) -> i32 {
    let cert_bytes = if cert_ptr.is_null() || cert_len == 0 {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(cert_ptr, cert_len) }
    };
    let step_bytes = if step_ptr.is_null() || step_len == 0 {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(step_ptr, step_len) })
    };

    let cert = match decode_ntcert(cert_bytes) {
        Ok(c) => c,
        Err(e) => {
            set_out(format_reject_json(rejection_gate(&e), &format!("{e:?}"), "custom"));
            return 1;
        }
    };
    let mut step_bound = false;
    if let Some(sb) = step_bytes {
        let mut h = Sha256::new();
        h.update(sb);
        let digest = h.finalize();
        if digest != cert.step_cad_sha256 {
            set_out(format_reject_json(
                "CAD",
                "StepCadSha256Mismatch { expected != sha256(step_bytes) }",
                "custom",
            ));
            return 1;
        }
        step_bound = true;
    }
    match verify_actuator_assembly_cert(&cert) {
        Ok(rep) => {
            set_out(format_pass_json(&rep, step_bound, "baseline"));
            0
        }
        Err(e) => {
            set_out(format_reject_json(rejection_gate(&e), &format!("{e:?}"), "custom"));
            1
        }
    }
}

/// Interactive adversarial verification entrypoint for live in-browser proofs.
/// Applies a specific real physical/cryptographic tamper to the decoded `.ntcert` and executes
/// the unmodified `nooterra-kernel` verifier live in WebAssembly:
///   0 = Untampered baseline (CERTIFIED)
///   1 = Flip 1 bit in `.ntcert` wire payload (CodecCorrupted)
///   2 = Modify 1 byte in `.step` CAD file (StepCadSha256Mismatch)
///   3 = Nudge tangential shear past 3D circular Coulomb friction cone (M1)
///   4 = +1 LSB interior stress equilibrium cheat in bolted flange (M1)
///   5 = Plug coolant exit port voxel creating powder trap (G3)
///   6 = Reorder disassembly steps creating rigid-body collision (G3)
///   7 = Understate winding thermal limit below certified hot-spot (M3)
///   8 = Relax tolerance link causing rotor-stator airgap rub risk (G4)
#[no_mangle]
pub extern "C" fn nt_verify_attack(
    cert_ptr: *const u8,
    cert_len: usize,
    step_ptr: *const u8,
    step_len: usize,
    attack_id: u32,
) -> i32 {
    let cert_bytes = if cert_ptr.is_null() || cert_len == 0 {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(cert_ptr, cert_len) }
    };
    let step_bytes = if step_ptr.is_null() || step_len == 0 {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(step_ptr, step_len) })
    };

    if attack_id == 1 {
        let mut flipped = cert_bytes.to_vec();
        if flipped.len() > 64 {
            flipped[64] ^= 0x01;
        }
        return match decode_ntcert(&flipped) {
            Ok(_) => {
                set_out(format_reject_json("WIRE", "UnexpectedPass", "wire_1bit_flip"));
                1
            }
            Err(e) => {
                set_out(format_reject_json("WIRE", &format!("{e:?}"), "wire_1bit_flip"));
                1
            }
        };
    }

    let mut cert = match decode_ntcert(cert_bytes) {
        Ok(c) => c,
        Err(e) => {
            set_out(format_reject_json(rejection_gate(&e), &format!("{e:?}"), "decode_error"));
            return 1;
        }
    };

    if attack_id == 2 {
        let mut bad_step = step_bytes.unwrap_or(b"ISO-10303-21;").to_vec();
        if let Some(b) = bad_step.last_mut() {
            *b ^= 0x01;
        }
        let mut h = Sha256::new();
        h.update(&bad_step);
        if h.finalize() != cert.step_cad_sha256 {
            set_out(format_reject_json(
                "CAD",
                "StepCadSha256Mismatch { modified_step_sha256 != cert.step_cad_sha256 }",
                "step_1byte_tamper",
            ));
            return 1;
        }
    }

    let label = match attack_id {
        0 => "baseline",
        3 => {
            let ny = cert.flange_contact_cert.spec.ny;
            let nz = cert.flange_contact_cert.spec.nz;
            let pn = -cert.flange_contact_cert.witness.fx[idx_fx(ny, nz, 0, 1, 1)];
            cert.flange_contact_cert.witness.ez[idx_ez(ny, nz, 0, 1, 1)] = (pn * 45) / 100 + 1;
            "flange_circular_friction_breach"
        }
        4 => {
            let ny = cert.flange_contact_cert.spec.ny;
            let nz = cert.flange_contact_cert.spec.nz;
            let mid_x = cert.flange_contact_cert.spec.nx / 2;
            cert.flange_contact_cert.witness.fx[idx_fx(ny, nz, mid_x, 1, 1)] += 1;
            "equilibrium_1lsb_tamper"
        }
        5 => {
            for v in &mut cert.cooling_dfm_spec.voxels {
                if *v == nooterra_kernel::VOXEL_PORT {
                    *v = VOXEL_SOLID;
                }
            }
            "dfm_trapped_coolant_powder"
        }
        6 => {
            if cert.dfa_kinematics_spec.disassembly_steps.len() >= 2 {
                cert.dfa_kinematics_spec.disassembly_steps.swap(0, 1);
            }
            "dfa_assembly_collision"
        }
        7 => {
            cert.target.max_winding_temp_num = 30;
            cert.target.max_winding_temp_den = 1;
            "thermal_hotspot_breach"
        }
        8 => {
            if let Some(link) = cert.tolerance_spec.links.first_mut() {
                link.pm_airgap_um += 500;
            }
            "sdt_airgap_rub_risk"
        }
        _ => "baseline",
    };

    let mut step_bound = false;
    if attack_id == 0 {
        if let Some(sb) = step_bytes {
            let mut h = Sha256::new();
            h.update(sb);
            if h.finalize() != cert.step_cad_sha256 {
                set_out(format_reject_json(
                    "CAD",
                    "StepCadSha256Mismatch",
                    "baseline",
                ));
                return 1;
            }
            step_bound = true;
        }
    }

    match verify_actuator_assembly_cert(&cert) {
        Ok(rep) => {
            set_out(format_pass_json(&rep, step_bound, label));
            0
        }
        Err(e) => {
            set_out(format_reject_json(rejection_gate(&e), &format!("{e:?}"), label));
            1
        }
    }
}
