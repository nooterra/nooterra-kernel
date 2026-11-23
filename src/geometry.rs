//! Pure-integer geometry, additive DFM, rigid-body assembly kinematics, and 3D SDT tolerance
//! verification (`G1`, `G3`, `G4`).

use alloc::vec;
use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};
use crate::sha256::Sha256;
use crate::thermal::GdtScaleBox3D;

pub const GEOMETRY_KERNEL_ID: &str = "nooterra-kernel/3.0.0 geom-dfm-dfa-gdt-exact";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryRejection {
    InvalidGrid,
    InvalidDenominator,
    InvalidRadius,
    MaskSizeMismatch,
    EmptyInnerShell,
    InnerNotSubsetOfOuter { i: usize, j: usize, k: usize },
    InnerShellProtrudesOutsideBrep { i: usize, j: usize, k: usize },
    OuterShellMissesBrep { i: usize, j: usize, k: usize },
    WallTooThin { axis: u8, i: usize, j: usize, k: usize, thickness: usize },
    UnsupportedOverhang { i: usize, j: usize, k: usize },
    TrappedCoolantPowder { i: usize, j: usize, k: usize },
    NoCoolantPort,
    InvalidAssemblyBox,
    InitialInterpenetration { part_a: u32, part_b: u32 },
    DuplicateOrMissingPartInSequence { part_id: u32 },
    InvalidDisassemblyDirection { step_idx: usize },
    AssemblyCollision { moving_part: u32, blocking_part: u32, step_idx: usize },
    InvalidToleranceSpec,
    AirgapRubRisk { min_airgap_num: i64, required_min_num: i64 },
}

/// Exact rational circular bolt hole `(cx, cy, radius)` divided by `CurvedAnnulusSpec::r_den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoltHole2D {
    pub cx_num: i64,
    pub cy_num: i64,
    pub radius_num: i64,
}

/// Exact rational hollow cylinder (annulus) with optional circular ISO bolt-circle holes.
/// Coordinates are in physical length units:
///   - Cell `(i, j, k)` spans `[i*dx, (i+1)*dx] x [j*dy, (j+1)*dy] x [k*dz, (k+1)*dz]` (`/ grid_den`).
///   - Cylinder axis is `(center_x_num, center_y_num) / r_den`, inner/outer radii `(r_in_num, r_out_num) / r_den`,
///     axial span `[z_min_num, z_max_num] / r_den`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurvedAnnulusSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub grid_den: u64,
    pub dx_num: u64,
    pub dy_num: u64,
    pub dz_num: u64,
    pub r_den: u64,
    pub center_x_num: i64,
    pub center_y_num: i64,
    pub r_in_num: i64,
    pub r_out_num: i64,
    pub z_min_num: i64,
    pub z_max_num: i64,
    pub bolt_holes: Vec<BoltHole2D>,
}

impl CurvedAnnulusSpec {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(GEOMETRY_KERNEL_ID.as_bytes());
        h.update(b"curved-annulus-v1");
        h.update_u64(self.nx as u64);
        h.update_u64(self.ny as u64);
        h.update_u64(self.nz as u64);
        h.update_u64(self.grid_den);
        h.update_u64(self.dx_num);
        h.update_u64(self.dy_num);
        h.update_u64(self.dz_num);
        h.update_u64(self.r_den);
        h.update_i64(self.center_x_num);
        h.update_i64(self.center_y_num);
        h.update_i64(self.r_in_num);
        h.update_i64(self.r_out_num);
        h.update_i64(self.z_min_num);
        h.update_i64(self.z_max_num);
        h.update_u64(self.bolt_holes.len() as u64);
        for b in &self.bolt_holes {
            h.update_i64(b.cx_num);
            h.update_i64(b.cy_num);
            h.update_i64(b.radius_num);
        }
        h.finalize()
    }
}

/// Witness containing inner inscribed shell (`Omega_in <= Omega_exact`) and
/// outer circumscribed shell (`Omega_out >= Omega_exact`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurvedSandwichWitness {
    pub in_mask: Vec<u8>,  // 1 = strictly inside curved B-Rep solid
    pub out_mask: Vec<u8>, // 1 = intersects or inside curved B-Rep solid
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurvedSandwichReport {
    pub spec_hash: [u8; 32],
    pub inner_cells: usize,
    pub outer_cells: usize,
    pub boundary_cells: usize,
    pub vol_inner: BigRat,
    pub vol_outer: BigRat,
    /// Exact rational volume sandwich tightness `(vol_outer - vol_inner) / vol_outer`.
    pub sandwich_gap_ratio: BigRat,
}

#[inline]
fn cell_dist_sq_bounds(x0: i128, x1: i128, y0: i128, y1: i128, cx: i128, cy: i128) -> (i128, i128) {
    let dx_min = if cx < x0 {
        x0 - cx
    } else if cx > x1 {
        cx - x1
    } else {
        0
    };
    let dy_min = if cy < y0 {
        y0 - cy
    } else if cy > y1 {
        cy - y1
    } else {
        0
    };
    let dx_max = (x0 - cx).abs().max((x1 - cx).abs());
    let dy_max = (y0 - cy).abs().max((y1 - cy).abs());
    (dx_min * dx_min + dy_min * dy_min, dx_max * dx_max + dy_max * dy_max)
}

/// Verifies exact integer inclusion `Omega_in <= Omega_exact <= Omega_out` for a hollow cylinder
/// with optional circular bolt holes (`G1`).
pub fn verify_curved_annulus_sandwich(
    spec: &CurvedAnnulusSpec,
    witness: &CurvedSandwichWitness,
) -> Result<CurvedSandwichReport, GeometryRejection> {
    if spec.nx == 0 || spec.ny == 0 || spec.nz == 0
        || spec.dx_num == 0 || spec.dy_num == 0 || spec.dz_num == 0
    {
        return Err(GeometryRejection::InvalidGrid);
    }
    if spec.grid_den == 0 || spec.r_den == 0 {
        return Err(GeometryRejection::InvalidDenominator);
    }
    if spec.r_in_num < 0 || spec.r_out_num <= spec.r_in_num || spec.z_max_num <= spec.z_min_num {
        return Err(GeometryRejection::InvalidRadius);
    }
    for b in &spec.bolt_holes {
        if b.radius_num <= 0 {
            return Err(GeometryRejection::InvalidRadius);
        }
    }
    let n_cells = spec.nx * spec.ny * spec.nz;
    if witness.in_mask.len() != n_cells || witness.out_mask.len() != n_cells {
        return Err(GeometryRejection::MaskSizeMismatch);
    }

    // Common integer coordinate scale: common_den = grid_den * r_den.
    let gd = spec.grid_den as i128;
    let rd = spec.r_den as i128;
    let dx = (spec.dx_num as i128) * rd;
    let dy = (spec.dy_num as i128) * rd;
    let dz = (spec.dz_num as i128) * rd;

    let cx = (spec.center_x_num as i128) * gd;
    let cy = (spec.center_y_num as i128) * gd;
    let r_in = (spec.r_in_num as i128) * gd;
    let r_out = (spec.r_out_num as i128) * gd;
    let r_in_sq = r_in * r_in;
    let r_out_sq = r_out * r_out;
    let z_min = (spec.z_min_num as i128) * gd;
    let z_max = (spec.z_max_num as i128) * gd;

    let holes: Vec<(i128, i128, i128)> = spec
        .bolt_holes
        .iter()
        .map(|b| {
            let br = (b.radius_num as i128) * gd;
            ((b.cx_num as i128) * gd, (b.cy_num as i128) * gd, br * br)
        })
        .collect();

    let mut inner_cells = 0usize;
    let mut outer_cells = 0usize;

    for i in 0..spec.nx {
        let x0 = (i as i128) * dx;
        let x1 = x0 + dx;
        for j in 0..spec.ny {
            let y0 = (j as i128) * dy;
            let y1 = y0 + dy;
            let (r_min_sq, r_max_sq) = cell_dist_sq_bounds(x0, x1, y0, y1, cx, cy);

            // Check exact bolt hole clearance/containment in (x, y)
            let mut strictly_outside_all_holes = true;
            let mut strictly_inside_some_hole = false;
            for &(bx, by, br_sq) in &holes {
                let (b_min_sq, b_max_sq) = cell_dist_sq_bounds(x0, x1, y0, y1, bx, by);
                if b_min_sq < br_sq {
                    strictly_outside_all_holes = false;
                }
                if b_max_sq <= br_sq {
                    strictly_inside_some_hole = true;
                }
            }

            for k in 0..spec.nz {
                let z0 = (k as i128) * dz;
                let z1 = z0 + dz;
                let cid = (i * spec.ny + j) * spec.nz + k;
                let in_v = witness.in_mask[cid] != 0;
                let out_v = witness.out_mask[cid] != 0;

                if in_v && !out_v {
                    return Err(GeometryRejection::InnerNotSubsetOfOuter { i, j, k });
                }

                let cell_strictly_inside = z0 >= z_min
                    && z1 <= z_max
                    && r_min_sq >= r_in_sq
                    && r_max_sq <= r_out_sq
                    && strictly_outside_all_holes;

                let cell_strictly_outside = z1 <= z_min
                    || z0 >= z_max
                    || r_max_sq <= r_in_sq
                    || r_min_sq >= r_out_sq
                    || strictly_inside_some_hole;

                if in_v {
                    if !cell_strictly_inside {
                        return Err(GeometryRejection::InnerShellProtrudesOutsideBrep { i, j, k });
                    }
                    inner_cells += 1;
                }
                if out_v {
                    outer_cells += 1;
                } else if !cell_strictly_outside {
                    return Err(GeometryRejection::OuterShellMissesBrep { i, j, k });
                }
            }
        }
    }

    if inner_cells == 0 {
        return Err(GeometryRejection::EmptyInnerShell);
    }

    let gdu = BigUint::from_u64(spec.grid_den);
    let vol_cell = BigRat::new(
        BigInt::from_i128((spec.dx_num as i128) * (spec.dy_num as i128) * (spec.dz_num as i128)),
        gdu.mul_u(&gdu).mul_u(&gdu),
    );
    let vol_inner = vol_cell.mul(&BigRat::from_i64(inner_cells as i64, 1));
    let vol_outer = vol_cell.mul(&BigRat::from_i64(outer_cells as i64, 1));
    let boundary_cells = outer_cells - inner_cells;
    let sandwich_gap_ratio = BigRat::from_i64(boundary_cells as i64, outer_cells as u64);

    Ok(CurvedSandwichReport {
        spec_hash: spec.spec_hash(),
        inner_cells,
        outer_cells,
        boundary_cells,
        vol_inner,
        vol_outer,
        sandwich_gap_ratio,
    })
}

pub const VOXEL_AIR: u8 = 0;
pub const VOXEL_SOLID: u8 = 1;
pub const VOXEL_COOLANT: u8 = 2;
pub const VOXEL_PORT: u8 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfmVoxelSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// `0 = ExternalAir`, `1 = SolidMetal`, `2 = InternalCoolantChannel`, `3 = CoolantInletOutletPort`.
    pub voxels: Vec<u8>,
    /// Minimum contiguous solid wall thickness (in cells) along x, y, z runs.
    pub min_wall_cells: usize,
    /// Require LPBF self-supporting <= 45 deg overhang cone from build plate `k = 0`.
    pub check_45deg_overhang: bool,
    /// Require EVERY `VOXEL_COOLANT` cell to have an unobstructed connected path to `VOXEL_PORT`.
    pub require_powder_escape: bool,
}

impl DfmVoxelSpec {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(GEOMETRY_KERNEL_ID.as_bytes());
        h.update(b"dfm-voxel-v1");
        h.update_u64(self.nx as u64);
        h.update_u64(self.ny as u64);
        h.update_u64(self.nz as u64);
        h.update_u64(self.min_wall_cells as u64);
        h.update(&[self.check_45deg_overhang as u8, self.require_powder_escape as u8]);
        h.update(&self.voxels);
        h.finalize()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfmReport {
    pub spec_hash: [u8; 32],
    pub solid_cells: usize,
    pub coolant_cells: usize,
    pub port_cells: usize,
    pub min_observed_wall_cells: usize,
}

/// Verifies minimum wall thickness, 45-degree LPBF self-support overhang, and BFS powder escape
/// from all internal coolant voxels to a port (`G3`).
pub fn verify_dfm_and_coolant_escape(spec: &DfmVoxelSpec) -> Result<DfmReport, GeometryRejection> {
    let (nx, ny, nz) = (spec.nx, spec.ny, spec.nz);
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(GeometryRejection::InvalidGrid);
    }
    let n_cells = nx * ny * nz;
    if spec.voxels.len() != n_cells {
        return Err(GeometryRejection::MaskSizeMismatch);
    }
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;

    let mut solid_cells = 0usize;
    let mut coolant_cells = 0usize;
    let mut port_cells = 0usize;
    for &v in &spec.voxels {
        match v {
            VOXEL_SOLID => solid_cells += 1,
            VOXEL_COOLANT => coolant_cells += 1,
            VOXEL_PORT => port_cells += 1,
            _ => {}
        }
    }

    let mut min_obs = usize::MAX;
    if spec.min_wall_cells > 0 {
        for j in 0..ny {
            for k in 0..nz {
                let mut i = 0;
                while i < nx {
                    if spec.voxels[idx(i, j, k)] == VOXEL_SOLID {
                        let start = i;
                        while i < nx && spec.voxels[idx(i, j, k)] == VOXEL_SOLID {
                            i += 1;
                        }
                        let run = i - start;
                        if run < min_obs {
                            min_obs = run;
                        }
                        if run < spec.min_wall_cells {
                            return Err(GeometryRejection::WallTooThin {
                                axis: 0, i: start, j, k, thickness: run,
                            });
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }
        // y-runs
        for i in 0..nx {
            for k in 0..nz {
                let mut j = 0;
                while j < ny {
                    if spec.voxels[idx(i, j, k)] == VOXEL_SOLID {
                        let start = j;
                        while j < ny && spec.voxels[idx(i, j, k)] == VOXEL_SOLID {
                            j += 1;
                        }
                        let run = j - start;
                        if run < min_obs {
                            min_obs = run;
                        }
                        if run < spec.min_wall_cells {
                            return Err(GeometryRejection::WallTooThin {
                                axis: 1, i, j: start, k, thickness: run,
                            });
                        }
                    } else {
                        j += 1;
                    }
                }
            }
        }
        // z-runs
        for i in 0..nx {
            for j in 0..ny {
                let mut k = 0;
                while k < nz {
                    if spec.voxels[idx(i, j, k)] == VOXEL_SOLID {
                        let start = k;
                        while k < nz && spec.voxels[idx(i, j, k)] == VOXEL_SOLID {
                            k += 1;
                        }
                        let run = k - start;
                        if run < min_obs {
                            min_obs = run;
                        }
                        if run < spec.min_wall_cells {
                            return Err(GeometryRejection::WallTooThin {
                                axis: 2, i, j, k: start, thickness: run,
                            });
                        }
                    } else {
                        k += 1;
                    }
                }
            }
        }
    }
    if min_obs == usize::MAX {
        min_obs = 0;
    }

    // 45-degree LPBF overhang: every solid cell at k > 0 needs support in the 3x3 patch at k - 1.
    if spec.check_45deg_overhang {
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz {
                    if spec.voxels[idx(i, j, k)] != VOXEL_SOLID {
                        continue;
                    }
                    let mut supported = false;
                    let i0 = i.saturating_sub(1);
                    let i1 = (i + 1).min(nx - 1);
                    let j0 = j.saturating_sub(1);
                    let j1 = (j + 1).min(ny - 1);
                    'sup: for si in i0..=i1 {
                        for sj in j0..=j1 {
                            if spec.voxels[idx(si, sj, k - 1)] == VOXEL_SOLID {
                                supported = true;
                                break 'sup;
                            }
                        }
                    }
                    if !supported {
                        return Err(GeometryRejection::UnsupportedOverhang { i, j, k });
                    }
                }
            }
        }
    }

    if spec.require_powder_escape && coolant_cells > 0 {
        if port_cells == 0 {
            return Err(GeometryRejection::NoCoolantPort);
        }
        let mut visited = vec![false; n_cells];
        let mut queue: Vec<(usize, usize, usize)> = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let id = idx(i, j, k);
                    if spec.voxels[id] == VOXEL_PORT {
                        visited[id] = true;
                        queue.push((i, j, k));
                    }
                }
            }
        }
        let mut head = 0usize;
        while head < queue.len() {
            let (ci, cj, ck) = queue[head];
            head += 1;
            let nbrs: [(Option<usize>, Option<usize>, Option<usize>); 6] = [
                (ci.checked_sub(1), Some(cj), Some(ck)),
                ((ci + 1 < nx).then_some(ci + 1), Some(cj), Some(ck)),
                (Some(ci), cj.checked_sub(1), Some(ck)),
                (Some(ci), (cj + 1 < ny).then_some(cj + 1), Some(ck)),
                (Some(ci), Some(cj), ck.checked_sub(1)),
                (Some(ci), Some(cj), (ck + 1 < nz).then_some(ck + 1)),
            ];
            for (ni, nj, nk) in nbrs {
                if let (Some(i2), Some(j2), Some(k2)) = (ni, nj, nk) {
                    let nid = idx(i2, j2, k2);
                    if !visited[nid]
                        && (spec.voxels[nid] == VOXEL_COOLANT || spec.voxels[nid] == VOXEL_PORT)
                    {
                        visited[nid] = true;
                        queue.push((i2, j2, k2));
                    }
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let id = idx(i, j, k);
                    if spec.voxels[id] == VOXEL_COOLANT && !visited[id] {
                        return Err(GeometryRejection::TrappedCoolantPowder { i, j, k });
                    }
                }
            }
        }
    }

    Ok(DfmReport {
        spec_hash: spec.spec_hash(),
        solid_cells,
        coolant_cells,
        port_cells,
        min_observed_wall_cells: min_obs,
    })
}

/// Exact integer axis-aligned bounding box `[x0, x1] x [y0, y1] x [z0, z1]` (e.g., micrometers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntBox3D {
    pub x0: i64,
    pub x1: i64,
    pub y0: i64,
    pub y1: i64,
    pub z0: i64,
    pub z1: i64,
}

impl IntBox3D {
    pub fn is_valid(&self) -> bool {
        self.x0 < self.x1 && self.y0 < self.y1 && self.z0 < self.z1
    }

    /// Strict interior overlap (`false` when boxes touch face-to-face at a mated datum).
    #[inline]
    pub fn overlaps(&self, o: &Self) -> bool {
        self.x0 < o.x1 && o.x0 < self.x1
            && self.y0 < o.y1 && o.y0 < self.y1
            && self.z0 < o.z1 && o.z0 < self.z1
    }

    /// Swept volume when extracting along `axis` (`0=x, 1=y, 2=z`) to `+inf` (`dir > 0`) or `-inf` (`dir < 0`).
    pub fn swept_to_inf(&self, axis: u8, dir: i8) -> Self {
        let mut s = *self;
        match (axis, dir > 0) {
            (0, true) => s.x1 = i64::MAX,
            (0, false) => s.x0 = i64::MIN,
            (1, true) => s.y1 = i64::MAX,
            (1, false) => s.y0 = i64::MIN,
            (2, true) => s.z1 = i64::MAX,
            (2, false) => s.z0 = i64::MIN,
            _ => {}
        }
        s
    }
}

/// Rigid component modeled as a union of integer axis-aligned sub-boxes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyPartGeom {
    pub part_id: u32,
    pub boxes: Vec<IntBox3D>,
}

/// Single disassembly step translating `part_id` along `axis` (`0=x, 1=y, 2=z`) in direction `dir` (`-1` or `+1`).
/// Reversing a verified disassembly sequence yields a collision-free assembly order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisassemblyStep {
    pub part_id: u32,
    pub axis: u8,
    pub dir: i8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyKinematicsSpec {
    pub parts: Vec<AssemblyPartGeom>,
    pub disassembly_steps: Vec<DisassemblyStep>,
}

impl AssemblyKinematicsSpec {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(GEOMETRY_KERNEL_ID.as_bytes());
        h.update(b"dfa-kinematics-v1");
        h.update_u64(self.parts.len() as u64);
        for p in &self.parts {
            h.update_u64(p.part_id as u64);
            h.update_u64(p.boxes.len() as u64);
            for b in &p.boxes {
                h.update_i64(b.x0); h.update_i64(b.x1);
                h.update_i64(b.y0); h.update_i64(b.y1);
                h.update_i64(b.z0); h.update_i64(b.z1);
            }
        }
        h.update_u64(self.disassembly_steps.len() as u64);
        for s in &self.disassembly_steps {
            h.update_u64(s.part_id as u64);
            h.update(&[s.axis, s.dir as u8]);
        }
        h.finalize()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyKinematicsReport {
    pub spec_hash: [u8; 32],
    pub num_parts: usize,
    pub num_steps_verified: usize,
}

/// Verifies zero initial interpenetration and collision-free rigid-body swept extraction (`G3-DFA`).
pub fn verify_rigid_assembly_sequence(
    spec: &AssemblyKinematicsSpec,
) -> Result<AssemblyKinematicsReport, GeometryRejection> {
    let n = spec.parts.len();
    if n == 0 || spec.disassembly_steps.len() != n {
        return Err(GeometryRejection::InvalidAssemblyBox);
    }
    for i in 0..n {
        if spec.parts[i].boxes.is_empty() {
            return Err(GeometryRejection::InvalidAssemblyBox);
        }
        for b in &spec.parts[i].boxes {
            if !b.is_valid() {
                return Err(GeometryRejection::InvalidAssemblyBox);
            }
        }
        for j in (i + 1)..n {
            if spec.parts[i].part_id == spec.parts[j].part_id {
                return Err(GeometryRejection::DuplicateOrMissingPartInSequence {
                    part_id: spec.parts[i].part_id,
                });
            }
            for ba in &spec.parts[i].boxes {
                for bb in &spec.parts[j].boxes {
                    if ba.overlaps(bb) {
                        return Err(GeometryRejection::InitialInterpenetration {
                            part_a: spec.parts[i].part_id,
                            part_b: spec.parts[j].part_id,
                        });
                    }
                }
            }
        }
    }

    let mut removed = vec![false; n];
    for (step_idx, step) in spec.disassembly_steps.iter().enumerate() {
        if step.axis > 2 || (step.dir != -1 && step.dir != 1) {
            return Err(GeometryRejection::InvalidDisassemblyDirection { step_idx });
        }
        let Some(p_idx) = spec.parts.iter().position(|p| p.part_id == step.part_id) else {
            return Err(GeometryRejection::DuplicateOrMissingPartInSequence {
                part_id: step.part_id,
            });
        };
        if removed[p_idx] {
            return Err(GeometryRejection::DuplicateOrMissingPartInSequence {
                part_id: step.part_id,
            });
        }
        for (other_idx, other) in spec.parts.iter().enumerate() {
            if other_idx == p_idx || removed[other_idx] {
                continue;
            }
            for bm in &spec.parts[p_idx].boxes {
                let swept = bm.swept_to_inf(step.axis, step.dir);
                for bo in &other.boxes {
                    if swept.overlaps(bo) {
                        return Err(GeometryRejection::AssemblyCollision {
                            moving_part: step.part_id,
                            blocking_part: other.part_id,
                            step_idx,
                        });
                    }
                }
            }
        }
        removed[p_idx] = true;
    }

    Ok(AssemblyKinematicsReport {
        spec_hash: spec.spec_hash(),
        num_parts: n,
        num_steps_verified: n,
    })
}

/// One datum or manufacturing tolerance link in the 3D assembly loop (micrometers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToleranceLinkUm {
    pub link_id: u32,
    /// Worst-case +/- positional/dimensional tolerance along x, y, z (micrometers >= 0).
    pub pm_x_um: u64,
    pub pm_y_um: u64,
    pub pm_z_um: u64,
    /// Contribution to stator-rotor airgap variation (+/- micrometers >= 0).
    pub pm_airgap_um: u64,
    /// Contribution to bolted flange flatness/contact gap (+ micrometers >= 0).
    pub pm_flange_gap_um: u64,
}

/// 3D Small Displacement Torsor (SDT) worst-case tolerance stackup specification.
/// Lengths are exact integers in micrometers (`1 mm = 1000 um`, `1 m = 1_000_000 um`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdtToleranceStackupSpec {
    pub nominal_x_um: u64,
    pub nominal_y_um: u64,
    pub nominal_z_um: u64,
    pub nominal_airgap_um: u64,
    pub min_safe_airgap_um: u64,
    pub flange_scale_um: u64, // flange characteristic length `1.0` in micrometers to normalize `gap_0`
    pub links: Vec<ToleranceLinkUm>,
}

impl SdtToleranceStackupSpec {
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(GEOMETRY_KERNEL_ID.as_bytes());
        h.update(b"sdt-tolerance-v1");
        h.update_u64(self.nominal_x_um);
        h.update_u64(self.nominal_y_um);
        h.update_u64(self.nominal_z_um);
        h.update_u64(self.nominal_airgap_um);
        h.update_u64(self.min_safe_airgap_um);
        h.update_u64(self.flange_scale_um);
        h.update_u64(self.links.len() as u64);
        for l in &self.links {
            h.update_u64(l.link_id as u64);
            h.update_u64(l.pm_x_um);
            h.update_u64(l.pm_y_um);
            h.update_u64(l.pm_z_um);
            h.update_u64(l.pm_airgap_um);
            h.update_u64(l.pm_flange_gap_um);
        }
        h.finalize()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdtToleranceReport {
    pub spec_hash: [u8; 32],
    pub total_pm_x_um: u64,
    pub total_pm_y_um: u64,
    pub total_pm_z_um: u64,
    pub total_pm_airgap_um: u64,
    pub min_airgap_um: i64,
    pub max_airgap_um: u64,
    /// Worst-case dimensionless flange contact gap `g_0_max = sum(pm_flange_gap_um) / flange_scale_um`.
    pub worst_flange_gap_norm: BigRat,
    /// Required 3D GD&T scale box for Maxwell EM (`z` axis scaled by airgap tolerance).
    pub required_em_gdt_box: GdtScaleBox3D,
    /// Required 3D GD&T scale box for thermal CHT dimensional tolerances.
    pub required_thermal_gdt_box: GdtScaleBox3D,
}

/// Verifies a 3D SDT tolerance chain (`G4`), returning worst-case airgap clearance, flange gap
/// `g_0_max`, and required `GdtScaleBox3D` envelopes for the `M1`, `M3`, and `M5` checks.
pub fn verify_sdt_tolerance_stackup(
    spec: &SdtToleranceStackupSpec,
) -> Result<SdtToleranceReport, GeometryRejection> {
    if spec.nominal_x_um == 0
        || spec.nominal_y_um == 0
        || spec.nominal_z_um == 0
        || spec.nominal_airgap_um == 0
        || spec.flange_scale_um == 0
        || spec.links.is_empty()
    {
        return Err(GeometryRejection::InvalidToleranceSpec);
    }

    let mut sx = 0u64;
    let mut sy = 0u64;
    let mut sz = 0u64;
    let mut sag = 0u64;
    let mut sfg = 0u64;
    for l in &spec.links {
        sx = sx.checked_add(l.pm_x_um).ok_or(GeometryRejection::InvalidToleranceSpec)?;
        sy = sy.checked_add(l.pm_y_um).ok_or(GeometryRejection::InvalidToleranceSpec)?;
        sz = sz.checked_add(l.pm_z_um).ok_or(GeometryRejection::InvalidToleranceSpec)?;
        sag = sag.checked_add(l.pm_airgap_um).ok_or(GeometryRejection::InvalidToleranceSpec)?;
        sfg = sfg.checked_add(l.pm_flange_gap_um).ok_or(GeometryRejection::InvalidToleranceSpec)?;
    }

    let min_airgap_um = (spec.nominal_airgap_um as i64) - (sag as i64);
    if min_airgap_um < (spec.min_safe_airgap_um as i64) {
        return Err(GeometryRejection::AirgapRubRisk {
            min_airgap_num: min_airgap_um,
            required_min_num: spec.min_safe_airgap_um as i64,
        });
    }
    let max_airgap_um = spec.nominal_airgap_um + sag;

    // Conservative integer ceiling to scale_den = 10_000 (0.01% resolution): ceil(pm * D / nom).
    const D: u64 = 10_000;
    let ceil_rel = |pm: u64, nom: u64| -> Result<u64, GeometryRejection> {
        let num = (pm as u128) * (D as u128) + (nom as u128) - 1;
        let q = (num / (nom as u128)) as u64;
        if q >= D {
            return Err(GeometryRejection::InvalidToleranceSpec);
        }
        Ok(q)
    };
    let rx = ceil_rel(sx, spec.nominal_x_um)?;
    let ry = ceil_rel(sy, spec.nominal_y_um)?;
    let rz = ceil_rel(sz, spec.nominal_z_um)?;
    let rag = ceil_rel(sag, spec.nominal_airgap_um)?;

    let required_em_gdt_box = GdtScaleBox3D {
        scale_den: D,
        ax_min: D - rx,
        ax_max: D + rx,
        ay_min: D - ry,
        ay_max: D + ry,
        az_min: D - rag,
        az_max: D + rag,
    };
    let required_thermal_gdt_box = GdtScaleBox3D {
        scale_den: D,
        ax_min: D - rx,
        ax_max: D + rx,
        ay_min: D - ry,
        ay_max: D + ry,
        az_min: D - rz,
        az_max: D + rz,
    };

    let worst_flange_gap_norm = BigRat::from_i64(sfg as i64, spec.flange_scale_um);

    Ok(SdtToleranceReport {
        spec_hash: spec.spec_hash(),
        total_pm_x_um: sx,
        total_pm_y_um: sy,
        total_pm_z_um: sz,
        total_pm_airgap_um: sag,
        min_airgap_um,
        max_airgap_um,
        worst_flange_gap_norm,
        required_em_gdt_box,
        required_thermal_gdt_box,
    })
}
