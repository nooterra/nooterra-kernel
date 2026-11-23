//! 3D simplicial strain-Riccati certificate verifier (`no_std`, `#![forbid(unsafe_code)]`).
//!
//! Independently checks:
//!   1. Positive orientation (`v6 > 0`) of every tetrahedron.
//!   2. Mesh boundary completeness (every face with incidence 1 must be classified in `spec`,
//!      and no interior face may be listed as a boundary face).
//!   3. Exact rational Neumann traction balance (`2 * g * n == -t`) and `3x3` Neumann boundary
//!      cone (`-H * n >= 0`) at all three vertices of each Neumann face.
//!   4. `10x10` strain-Riccati PSD cones at all `4 * T` tetrahedral corners across each material
//!      and frequency uncertainty corner via the two-tier PSD verifier.
//!   5. Exact rational lower bound `L = int_Omega c dV` (giving compliance upper bound
//!      `C_comp <= -2 * L`, or frequency lower bound `omega_1 >= omega_min` when `L >= 0`).
//!   6. Compositional interface contract matching (`verify_composed`) with opposite normals and
//!      identical rational `(g, H)` traces across matched faces.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};
use crate::cert::{
    BoundaryFaceKind, MaterialCorner, SimplicialCert3D, Spec3D, Witness3D, G_IDX, H_IDX, STRAIN_PAIR,
};
use crate::psd::{exact_bareiss_int_is_psd, verify_psd_i128_scaled, PsdMode, PsdOutcome};
use crate::KERNEL_ID;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    InvalidDenominator,
    EmptyMesh,
    EmptyUncertaintyBox,
    InvalidMaterialParams,
    WitnessSizeMismatch,
    VertexIndexOutOfBounds { tet: usize },
    NonPositiveTetVolume { tet: usize },
    InvalidBoundaryFace { face: usize },
    DuplicateBoundaryFace { face: usize },
    UnclassifiedBoundaryFace { verts: [u32; 3] },
    UnsupportedNonAxisNeumannTraction { face: usize },
    NeumannTractionMismatch { face: usize, vert: u32, comp: usize },
    NeumannConeNotPsd { face: usize, vert: u32 },
    ElementConeNotPsd { tet: usize, corner: usize, mat_corner: usize },
    InterfaceCountMismatch,
    InterfaceFaceUnmatched { tag: u32 },
    InterfaceNormalNotOpposite { tag: u32 },
    InterfaceFluxMismatch { tag: u32 },
}

/// Telemetry & exact certified rational results for a proven 3D certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofReport {
    pub spec_hash: [u8; 32],
    pub num_tets: usize,
    pub num_mat_corners: usize,
    pub total_psd_checks: usize,
    pub rump_fast_passes: usize,
    pub exact_bareiss_passes: usize,
    /// Exact rational total lower bound `L = int_Omega c dV`.
    pub lower_bound_l: BigRat,
    /// Exact rational compliance upper bound `C_upper = -2 * L` (when `t != 0, omega = 0`).
    pub compliance_upper: BigRat,
    /// Number of classified compositional interface faces.
    pub num_interface_faces: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposedProofReport {
    pub report_a: ProofReport,
    pub report_b: ProofReport,
    pub matched_interface_faces: usize,
    pub total_lower_bound_l: BigRat,
    pub total_compliance_upper: BigRat,
}

#[inline]
fn sub3(a: [i64; 3], b: [i64; 3]) -> [i128; 3] {
    [
        (a[0] as i128) - (b[0] as i128),
        (a[1] as i128) - (b[1] as i128),
        (a[2] as i128) - (b[2] as i128),
    ]
}

#[inline]
fn cross3(u: [i128; 3], v: [i128; 3]) -> Option<[i128; 3]> {
    let x = u[1].checked_mul(v[2])?.checked_sub(u[2].checked_mul(v[1])?)?;
    let y = u[2].checked_mul(v[0])?.checked_sub(u[0].checked_mul(v[2])?)?;
    let z = u[0].checked_mul(v[1])?.checked_sub(u[1].checked_mul(v[0])?)?;
    Some([x, y, z])
}

#[inline]
fn dot3(u: [i128; 3], v: [i128; 3]) -> Option<i128> {
    let p0 = u[0].checked_mul(v[0])?;
    let p1 = u[1].checked_mul(v[1])?;
    let p2 = u[2].checked_mul(v[2])?;
    p0.checked_add(p1)?.checked_add(p2)
}

#[inline]
fn sorted_tri(v: [u32; 3]) -> [u32; 3] {
    let mut s = v;
    if s[0] > s[1] {
        s.swap(0, 1);
    }
    if s[1] > s[2] {
        s.swap(1, 2);
    }
    if s[0] > s[1] {
        s.swap(0, 1);
    }
    s
}

/// Verifies a single 10x10 tet-corner cone using exact `i128` scaling (falling back to exact
/// `BigInt` if any intermediate product would overflow `i128`).
fn check_tet_corner_10x10(
    v6: i128,
    coord_den: i128,
    flux_den: i128,
    mat_den: i128,
    mc: MaterialCorner,
    c_num: i64,
    g_corner: [i64; 6],
    h_corner: [i64; 10],
    div_g_raw: [i128; 3],
    div_h_raw: [[i128; 3]; 3],
    mode: PsdMode,
) -> PsdOutcome {
    if let Some(m128) = assemble_10x10_i128(
        v6, coord_den, flux_den, mat_den, mc, c_num, g_corner, h_corner, div_g_raw, div_h_raw,
    ) {
        return verify_psd_i128_scaled(&m128, 10, mode);
    }
    let mut b = assemble_10x10_bigint(
        v6, coord_den, flux_den, mat_den, mc, c_num, g_corner, h_corner, div_g_raw, div_h_raw,
    );
    if exact_bareiss_int_is_psd(&mut b, 10) {
        PsdOutcome::ProvenExactBareiss
    } else {
        PsdOutcome::Rejected
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_10x10_i128(
    v6: i128,
    coord_den: i128,
    flux_den: i128,
    mat_den: i128,
    mc: MaterialCorner,
    c_num: i64,
    g_corner: [i64; 6],
    h_corner: [i64; 10],
    div_g_raw: [i128; 3],
    div_h_raw: [[i128; 3]; 3],
) -> Option<[i128; 100]> {
    let s_local = 2i128.checked_mul(v6)?.checked_mul(mat_den)?;
    let s_div = 2i128.checked_mul(coord_den)?.checked_mul(mat_den)?;
    let s_mat = v6.checked_mul(flux_den)?;
    let s_w2 = 2i128.checked_mul(s_mat)?;

    let mut m = [0i128; 100];
    // (0, 0): -c
    m[0] = (-(c_num as i128)).checked_mul(s_local)?;
    // (0, 1..4): div g
    for i in 0..3 {
        let v = div_g_raw[i].checked_mul(s_div)?;
        m[1 + i] = v;
        m[(1 + i) * 10] = v;
    }
    // (1..4, 1..4): div H - omega^2*rho*I
    let w2_term = (mc.omega2_rho_num as i128).checked_mul(s_w2)?;
    for i in 0..3 {
        for j in i..3 {
            let mut v = div_h_raw[i][j].checked_mul(s_div)?;
            if i == j {
                v = v.checked_sub(w2_term)?;
            }
            m[(1 + i) * 10 + (1 + j)] = v;
            m[(1 + j) * 10 + (1 + i)] = v;
        }
    }
    // (0, 4..10): corner g
    for p in 0..6 {
        let v = (g_corner[p] as i128).checked_mul(s_local)?;
        m[4 + p] = v;
        m[(4 + p) * 10] = v;
    }
    // (1..4, 4..10): corner H
    for i in 0..3 {
        for p in 0..6 {
            let (j, k) = STRAIN_PAIR[p];
            let v = (h_corner[H_IDX[i][j][k]] as i128).checked_mul(s_local)?;
            m[(1 + i) * 10 + (4 + p)] = v;
            m[(4 + p) * 10 + (1 + i)] = v;
        }
    }
    // (4..10, 4..10): 6x6 engineering-strain elasticity block D_6 = 0.5 * C(E)
    let diag_n = (2i128.checked_mul(mc.mu_num as i128)?.checked_add(mc.lam_num as i128)?)
        .checked_mul(s_mat)?;
    let off_n = (mc.lam_num as i128).checked_mul(s_mat)?;
    let diag_s = (mc.mu_num as i128).checked_mul(s_mat)?;
    for p in 0..3 {
        m[(4 + p) * 10 + (4 + p)] = diag_n;
    }
    for (p, q) in [(0, 1), (0, 2), (1, 2)] {
        m[(4 + p) * 10 + (4 + q)] = off_n;
        m[(4 + q) * 10 + (4 + p)] = off_n;
    }
    for p in 3..6 {
        m[(4 + p) * 10 + (4 + p)] = diag_s;
    }
    Some(m)
}

#[allow(clippy::too_many_arguments)]
fn assemble_10x10_bigint(
    v6: i128,
    coord_den: i128,
    flux_den: i128,
    mat_den: i128,
    mc: MaterialCorner,
    c_num: i64,
    g_corner: [i64; 6],
    h_corner: [i64; 10],
    div_g_raw: [i128; 3],
    div_h_raw: [[i128; 3]; 3],
) -> Vec<BigInt> {
    let bi = BigInt::from_i128;
    let s_local = bi(2).mul(&bi(v6)).mul(&bi(mat_den));
    let s_div = bi(2).mul(&bi(coord_den)).mul(&bi(mat_den));
    let s_mat = bi(v6).mul(&bi(flux_den));
    let s_w2 = bi(2).mul(&s_mat);

    let mut m = vec![BigInt::zero(); 100];
    m[0] = bi(-(c_num as i128)).mul(&s_local);
    for i in 0..3 {
        let v = bi(div_g_raw[i]).mul(&s_div);
        m[1 + i] = v.clone();
        m[(1 + i) * 10] = v;
    }
    let w2_term = bi(mc.omega2_rho_num as i128).mul(&s_w2);
    for i in 0..3 {
        for j in i..3 {
            let mut v = bi(div_h_raw[i][j]).mul(&s_div);
            if i == j {
                v = v.sub(&w2_term);
            }
            m[(1 + i) * 10 + (1 + j)] = v.clone();
            m[(1 + j) * 10 + (1 + i)] = v;
        }
    }
    for p in 0..6 {
        let v = bi(g_corner[p] as i128).mul(&s_local);
        m[4 + p] = v.clone();
        m[(4 + p) * 10] = v;
    }
    for i in 0..3 {
        for p in 0..6 {
            let (j, k) = STRAIN_PAIR[p];
            let v = bi(h_corner[H_IDX[i][j][k]] as i128).mul(&s_local);
            m[(1 + i) * 10 + (4 + p)] = v.clone();
            m[(4 + p) * 10 + (1 + i)] = v;
        }
    }
    let diag_n = bi(2).mul(&bi(mc.mu_num as i128)).add(&bi(mc.lam_num as i128)).mul(&s_mat);
    let off_n = bi(mc.lam_num as i128).mul(&s_mat);
    let diag_s = bi(mc.mu_num as i128).mul(&s_mat);
    for p in 0..3 {
        m[(4 + p) * 10 + (4 + p)] = diag_n.clone();
    }
    for (p, q) in [(0, 1), (0, 2), (1, 2)] {
        m[(4 + p) * 10 + (4 + q)] = off_n.clone();
        m[(4 + q) * 10 + (4 + p)] = off_n.clone();
    }
    for p in 3..6 {
        m[(4 + p) * 10 + (4 + p)] = diag_s.clone();
    }
    m
}

/// Helper: computes `div_g_raw` and `div_h_raw` on a single tet `(v0, v1, v2, v3)` with
/// integer outward-area normals `n[0..4]`.
fn compute_tet_divs(
    tet: [u32; 4],
    n: [[i128; 3]; 4],
    w: &Witness3D,
) -> Option<([i128; 3], [[i128; 3]; 3])> {
    let mut div_g = [0i128; 3];
    let mut div_h = [[0i128; 3]; 3];
    for r in 0..4 {
        let nd = tet[r] as usize;
        let gr = &w.g_nodes[nd];
        let hr = &w.h_nodes[nd];
        for d in 0..3 {
            let nd_val = n[r][d];
            if nd_val == 0 {
                continue;
            }
            for i in 0..3 {
                let g_id = gr[G_IDX[i][d]] as i128;
                div_g[i] = div_g[i].checked_add(nd_val.checked_mul(g_id)?)?;
            }
            for i in 0..3 {
                for j in i..3 {
                    let h_dij = hr[H_IDX[d][i][j]] as i128;
                    div_h[i][j] = div_h[i][j].checked_add(nd_val.checked_mul(h_dij)?)?;
                }
            }
        }
    }
    for i in 0..3 {
        for j in 0..i {
            div_h[i][j] = div_h[j][i];
        }
    }
    Some((div_g, div_h))
}

/// Precomputed tet geometry for parallel or sequential verification.
#[derive(Clone, Copy)]
struct TetGeom {
    v6: i128,
    normals: [[i128; 3]; 4],
}

/// Validates mesh topology, boundary faces, Neumann BCs, and extracts precomputed tet geometries.
fn validate_mesh_and_boundaries(
    spec: &Spec3D,
    witness: &Witness3D,
    mode: PsdMode,
) -> Result<(Vec<TetGeom>, usize, usize, usize), Rejection> {
    if spec.coord_den == 0 || spec.mat_den == 0 || witness.flux_den == 0 {
        return Err(Rejection::InvalidDenominator);
    }
    if spec.verts.is_empty() || spec.tets.is_empty() {
        return Err(Rejection::EmptyMesh);
    }
    if spec.material_corners.is_empty() {
        return Err(Rejection::EmptyUncertaintyBox);
    }
    for mc in &spec.material_corners {
        // Strict elasticity positivity: mu > 0 and 2*mu + 3*lam > 0 (bulk modulus > 0)
        if mc.mu_num <= 0 || (2i128 * (mc.mu_num as i128) + 3i128 * (mc.lam_num as i128)) <= 0 {
            return Err(Rejection::InvalidMaterialParams);
        }
        if mc.omega2_rho_num < 0 {
            return Err(Rejection::InvalidMaterialParams);
        }
    }
    let nn = spec.verts.len();
    let nt = spec.tets.len();
    if witness.g_nodes.len() != nn
        || witness.h_nodes.len() != nn
        || witness.c_corners.len() != 4 * nt
    {
        return Err(Rejection::WitnessSizeMismatch);
    }

    let mut geoms = Vec::with_capacity(nt);
    let mut face_count: BTreeMap<[u32; 3], (u32, [u32; 3])> = BTreeMap::new();
    for (ti, &t) in spec.tets.iter().enumerate() {
        for &v in &t {
            if (v as usize) >= nn {
                return Err(Rejection::VertexIndexOutOfBounds { tet: ti });
            }
        }
        let p0 = spec.verts[t[0] as usize];
        let p1 = spec.verts[t[1] as usize];
        let p2 = spec.verts[t[2] as usize];
        let p3 = spec.verts[t[3] as usize];
        let e1 = sub3(p1, p0);
        let e2 = sub3(p2, p0);
        let e3 = sub3(p3, p0);
        let n1 = cross3(e2, e3).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        let n2 = cross3(e3, e1).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        let n3 = cross3(e1, e2).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        let v6 = dot3(e1, n1).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        if v6 <= 0 {
            return Err(Rejection::NonPositiveTetVolume { tet: ti });
        }
        let n0 = [
            -(n1[0] + n2[0] + n3[0]),
            -(n1[1] + n2[1] + n3[1]),
            -(n1[2] + n2[2] + n3[2]),
        ];
        geoms.push(TetGeom { v6, normals: [n0, n1, n2, n3] });

        // Outward faces of positively-oriented tet (a, b, c, d): (a,c,b), (a,b,d), (b,c,d), (c,a,d)
        let [a, b, c, d] = t;
        for f in [[a, c, b], [a, b, d], [b, c, d], [c, a, d]] {
            let key = sorted_tri(f);
            let entry = face_count.entry(key).or_insert((0, f));
            entry.0 += 1;
            entry.1 = f;
        }
    }

    // Exact set of true boundary faces (incidence == 1) with outward orientation
    let mut true_bnd: BTreeMap<[u32; 3], ([u32; 3], bool)> = BTreeMap::new();
    for (key, (cnt, oriented)) in face_count {
        if cnt == 1 {
            true_bnd.insert(key, (oriented, false));
        }
    }

    let mut bnd_rump = 0usize;
    let mut bnd_exact = 0usize;
    let mut num_iface = 0usize;

    for (fi, bf) in spec.boundary_faces.iter().enumerate() {
        let key = sorted_tri(bf.verts);
        let (oriented, seen) = match true_bnd.get_mut(&key) {
            Some(entry) => entry,
            None => return Err(Rejection::InvalidBoundaryFace { face: fi }),
        };
        if *seen {
            return Err(Rejection::DuplicateBoundaryFace { face: fi });
        }
        *seen = true;
        let [v0, v1, v2] = *oriented;
        let p0 = spec.verts[v0 as usize];
        let p1 = spec.verts[v1 as usize];
        let p2 = spec.verts[v2 as usize];
        let nf = cross3(sub3(p1, p0), sub3(p2, p0)).ok_or(Rejection::InvalidBoundaryFace { face: fi })?;
        if nf == [0, 0, 0] {
            return Err(Rejection::InvalidBoundaryFace { face: fi });
        }

        match bf.kind {
            BoundaryFaceKind::Dirichlet => {}
            BoundaryFaceKind::Interface { .. } => {
                num_iface += 1;
            }
            BoundaryFaceKind::Contact { .. } => {
                // Contact BCs in tet Riccati mode are validated by dedicated contact checkers
                return Err(Rejection::InvalidBoundaryFace { face: fi });
            }
            BoundaryFaceKind::Neumann { traction_num, traction_den } => {
                if traction_den == 0 {
                    return Err(Rejection::InvalidDenominator);
                }
                let is_zero_t = traction_num == [0, 0, 0];
                let nz_axes: Vec<usize> = (0..3).filter(|&d| nf[d] != 0).collect();

                for &nd in &[v0, v1, v2] {
                    let g = &witness.g_nodes[nd as usize];
                    let h = &witness.h_nodes[nd as usize];

                    // Exact traction balance: 2 * g * n == -t
                    if is_zero_t {
                        // Exact integer g * nf == 0 holds for arbitrary triangle orientation without sqrt.
                        for i in 0..3 {
                            let mut acc = 0i128;
                            for d in 0..3 {
                                acc += (g[G_IDX[i][d]] as i128) * nf[d];
                            }
                            if acc != 0 {
                                return Err(Rejection::NeumannTractionMismatch {
                                    face: fi,
                                    vert: nd,
                                    comp: i,
                                });
                            }
                        }
                    } else if nz_axes.len() == 1 {
                        let d = nz_axes[0];
                        let sgn: i128 = if nf[d] > 0 { 1 } else { -1 };
                        for i in 0..3 {
                            let lhs = 2i128 * sgn * (g[G_IDX[i][d]] as i128) * (traction_den as i128);
                            let rhs = -(traction_num[i] as i128) * (witness.flux_den as i128);
                            if lhs != rhs {
                                return Err(Rejection::NeumannTractionMismatch {
                                    face: fi,
                                    vert: nd,
                                    comp: i,
                                });
                            }
                        }
                    } else {
                        return Err(Rejection::UnsupportedNonAxisNeumannTraction { face: fi });
                    }

                    // 3x3 Neumann boundary cone: -(H * nf) >= 0 for arbitrary face orientation.
                    let mut h3 = [0i128; 9];
                    for i in 0..3 {
                        for j in i..3 {
                            let mut acc = 0i128;
                            for d in 0..3 {
                                acc -= nf[d] * (h[H_IDX[d][i][j]] as i128);
                            }
                            h3[i * 3 + j] = acc;
                            h3[j * 3 + i] = acc;
                        }
                    }
                    match verify_psd_i128_scaled(&h3, 3, mode) {
                        PsdOutcome::ProvenFastRump => bnd_rump += 1,
                        PsdOutcome::ProvenExactBareiss => bnd_exact += 1,
                        PsdOutcome::Rejected => {
                            return Err(Rejection::NeumannConeNotPsd { face: fi, vert: nd });
                        }
                    }
                }
            }
        }
    }

    for (_key, (oriented, seen)) in true_bnd {
        if !seen {
            return Err(Rejection::UnclassifiedBoundaryFace { verts: oriented });
        }
    }

    Ok((geoms, bnd_rump, bnd_exact, num_iface))
}

/// Sequential (`no_std`) verifier for a 3D simplicial certificate.
pub fn verify_cert(cert: &SimplicialCert3D, mode: PsdMode) -> Result<ProofReport, Rejection> {
    verify_cert_range_hook(cert, mode, |spec, witness, geoms, n_tets| {
        let mut rump_cnt = 0usize;
        let mut exact_cnt = 0usize;
        for ti in 0..n_tets {
            let (r, e) = verify_single_tet(spec, witness, geoms[ti], ti, mode)?;
            rump_cnt += r;
            exact_cnt += e;
        }
        Ok((rump_cnt, exact_cnt))
    })
}

/// Verifies all `4 * material_corners.len()` PSD cones for a range of tetrahedra so `std`
/// callers can schedule verification across threads outside the `no_std` crate.
pub fn verify_tet_batch(
    cert: &SimplicialCert3D,
    mode: PsdMode,
    tet_indices: core::ops::Range<usize>,
) -> Result<(usize, usize), Rejection> {
    let spec = &cert.spec;
    let w = &cert.witness;
    let mut rump_cnt = 0usize;
    let mut exact_cnt = 0usize;
    for ti in tet_indices {
        let t = spec.tets[ti];
        let p0 = spec.verts[t[0] as usize];
        let p1 = spec.verts[t[1] as usize];
        let p2 = spec.verts[t[2] as usize];
        let p3 = spec.verts[t[3] as usize];
        let e1 = sub3(p1, p0);
        let e2 = sub3(p2, p0);
        let e3 = sub3(p3, p0);
        let n1 = cross3(e2, e3).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        let n2 = cross3(e3, e1).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        let n3 = cross3(e1, e2).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        let v6 = dot3(e1, n1).ok_or(Rejection::NonPositiveTetVolume { tet: ti })?;
        if v6 <= 0 {
            return Err(Rejection::NonPositiveTetVolume { tet: ti });
        }
        let n0 = [
            -(n1[0] + n2[0] + n3[0]),
            -(n1[1] + n2[1] + n3[1]),
            -(n1[2] + n2[2] + n3[2]),
        ];
        let geom = TetGeom { v6, normals: [n0, n1, n2, n3] };
        let (r, e) = verify_single_tet(spec, w, geom, ti, mode)?;
        rump_cnt += r;
        exact_cnt += e;
    }
    Ok((rump_cnt, exact_cnt))
}

fn verify_single_tet(
    spec: &Spec3D,
    w: &Witness3D,
    geom: TetGeom,
    ti: usize,
    mode: PsdMode,
) -> Result<(usize, usize), Rejection> {
    let t = spec.tets[ti];
    let (div_g_raw, div_h_raw) = compute_tet_divs(t, geom.normals, w)
        .ok_or(Rejection::ElementConeNotPsd { tet: ti, corner: 0, mat_corner: 0 })?;
    let mut rump = 0usize;
    let mut exact = 0usize;
    for m in 0..4 {
        let nd = t[m] as usize;
        let c_num = w.c_corners[4 * ti + m];
        let g_c = w.g_nodes[nd];
        let h_c = w.h_nodes[nd];
        for (mi, &mc) in spec.material_corners.iter().enumerate() {
            match check_tet_corner_10x10(
                geom.v6,
                spec.coord_den as i128,
                w.flux_den as i128,
                spec.mat_den as i128,
                mc,
                c_num,
                g_c,
                h_c,
                div_g_raw,
                div_h_raw,
                mode,
            ) {
                PsdOutcome::ProvenFastRump => rump += 1,
                PsdOutcome::ProvenExactBareiss => exact += 1,
                PsdOutcome::Rejected => {
                    return Err(Rejection::ElementConeNotPsd {
                        tet: ti,
                        corner: m,
                        mat_corner: mi,
                    });
                }
            }
        }
    }
    Ok((rump, exact))
}

/// Core hook enabling external parallel scheduling (e.g., `std::thread::scope`) while keeping
/// all math, boundary checks, exact bound integration, and spec hashing inside `no_std` kernel.
pub fn verify_cert_with_executor<F>(
    cert: &SimplicialCert3D,
    mode: PsdMode,
    run_tets: F,
) -> Result<ProofReport, Rejection>
where
    F: FnOnce(&SimplicialCert3D, PsdMode, usize) -> Result<(usize, usize), Rejection>,
{
    verify_cert_range_hook(cert, mode, |_spec, _w, _geoms, nt| run_tets(cert, mode, nt))
}

fn verify_cert_range_hook<F>(
    cert: &SimplicialCert3D,
    mode: PsdMode,
    run_tets: F,
) -> Result<ProofReport, Rejection>
where
    F: FnOnce(&Spec3D, &Witness3D, &[TetGeom], usize) -> Result<(usize, usize), Rejection>,
{
    let (geoms, bnd_rump, bnd_exact, num_iface) =
        validate_mesh_and_boundaries(&cert.spec, &cert.witness, mode)?;
    let nt = geoms.len();
    let (tet_rump, tet_exact) = run_tets(&cert.spec, &cert.witness, &geoms, nt)?;

    // Exact arbitrary-precision lower bound L = (sum_t v6_t * sum_m c_{t,m}) / (24 * coord_den^3 * flux_den)
    let mut num_acc = BigInt::zero();
    for (ti, geom) in geoms.iter().enumerate() {
        let c_sum = (cert.witness.c_corners[4 * ti] as i128)
            + (cert.witness.c_corners[4 * ti + 1] as i128)
            + (cert.witness.c_corners[4 * ti + 2] as i128)
            + (cert.witness.c_corners[4 * ti + 3] as i128);
        let term = BigInt::from_i128(geom.v6).mul(&BigInt::from_i128(c_sum));
        num_acc = num_acc.add(&term);
    }
    let cd = BigUint::from_u64(cert.spec.coord_den);
    let cd3 = cd.mul_u(&cd).mul_u(&cd);
    let den = BigUint::from_u64(24)
        .mul_u(&cd3)
        .mul_u(&BigUint::from_u64(cert.witness.flux_den));
    let lower_bound_l = BigRat::new(num_acc, den);
    let compliance_upper = lower_bound_l.mul(&BigRat::from_i64(-2, 1));

    let total_psd = bnd_rump + bnd_exact + tet_rump + tet_exact;
    Ok(ProofReport {
        spec_hash: cert.spec.spec_hash(KERNEL_ID),
        num_tets: nt,
        num_mat_corners: cert.spec.material_corners.len(),
        total_psd_checks: total_psd,
        rump_fast_passes: bnd_rump + tet_rump,
        exact_bareiss_passes: bnd_exact + tet_exact,
        lower_bound_l,
        compliance_upper,
        num_interface_faces: num_iface,
    })
}

/// Extracts interface face records keyed by `(tag, sorted physical vertex coords)` so
/// subdomains A and B match independently of local vertex numbering.
fn extract_interface_faces(
    cert: &SimplicialCert3D,
) -> Result<BTreeMap<(u32, [[BigRat; 3]; 3]), ([BigRat; 3], [[BigRat; 6]; 3], [[BigRat; 10]; 3])>, Rejection>
{
    let mut face_count: BTreeMap<[u32; 3], (u32, [u32; 3])> = BTreeMap::new();
    for &[a, b, c, d] in &cert.spec.tets {
        for f in [[a, c, b], [a, b, d], [b, c, d], [c, a, d]] {
            let key = sorted_tri(f);
            let e = face_count.entry(key).or_insert((0, f));
            e.0 += 1;
            e.1 = f;
        }
    }
    let cd = cert.spec.coord_den;
    let fd = cert.witness.flux_den;
    let mut out = BTreeMap::new();
    for bf in &cert.spec.boundary_faces {
        if let BoundaryFaceKind::Interface { tag } = bf.kind {
            let key = sorted_tri(bf.verts);
            let (_, oriented) = face_count.get(&key).ok_or(Rejection::InterfaceFaceUnmatched { tag })?;
            let [v0, v1, v2] = *oriented;
            let p0 = cert.spec.verts[v0 as usize];
            let p1 = cert.spec.verts[v1 as usize];
            let p2 = cert.spec.verts[v2 as usize];
            let nf = cross3(sub3(p1, p0), sub3(p2, p0))
                .ok_or(Rejection::InterfaceFaceUnmatched { tag })?;
            let cd2 = (cd as u128) * (cd as u128);
            let normal_q = [
                BigRat::from_i128(nf[0], cd2),
                BigRat::from_i128(nf[1], cd2),
                BigRat::from_i128(nf[2], cd2),
            ];
            // Sort the 3 vertices canonically by physical coordinates so A and B match 1-to-1
            let mut v_items: [([BigRat; 3], [BigRat; 6], [BigRat; 10]); 3] = core::array::from_fn(|k| {
                let v = oriented[k] as usize;
                let coords = core::array::from_fn(|d| BigRat::from_i64(cert.spec.verts[v][d], cd));
                let gq = core::array::from_fn(|p| BigRat::from_i64(cert.witness.g_nodes[v][p], fd));
                let hq = core::array::from_fn(|p| BigRat::from_i64(cert.witness.h_nodes[v][p], fd));
                (coords, gq, hq)
            });
            v_items.sort_by(|a, b| cmp_vec3(&a.0, &b.0));
            let coord_key = [v_items[0].0.clone(), v_items[1].0.clone(), v_items[2].0.clone()];
            let g_arr = [v_items[0].1.clone(), v_items[1].1.clone(), v_items[2].1.clone()];
            let h_arr = [v_items[0].2.clone(), v_items[1].2.clone(), v_items[2].2.clone()];
            out.insert((tag, coord_key), (normal_q, g_arr, h_arr));
        }
    }
    Ok(out)
}

fn cmp_vec3(a: &[BigRat; 3], b: &[BigRat; 3]) -> core::cmp::Ordering {
    for d in 0..3 {
        let c = a[d].cmp_r(&b[d]);
        if c != core::cmp::Ordering::Equal {
            return c;
        }
    }
    core::cmp::Ordering::Equal
}

/// Verifies subdomains A and B independently and checks that every interface face matches
/// 1-to-1 with opposite outward normals (`n_A == -n_B`) and identical rational flux traces
/// (`g_A == g_B`, `H_A == H_B`), establishing `L_total = L_A + L_B`.
pub fn verify_composed(
    cert_a: &SimplicialCert3D,
    cert_b: &SimplicialCert3D,
    mode: PsdMode,
) -> Result<ComposedProofReport, Rejection> {
    let report_a = verify_cert(cert_a, mode)?;
    let report_b = verify_cert(cert_b, mode)?;
    let if_a = extract_interface_faces(cert_a)?;
    let if_b = extract_interface_faces(cert_b)?;
    if if_a.len() != if_b.len() {
        return Err(Rejection::InterfaceCountMismatch);
    }
    for ((tag, coords), (na, ga, ha)) in &if_a {
        let (nb, gb, hb) = if_b
            .get(&(*tag, coords.clone()))
            .ok_or(Rejection::InterfaceFaceUnmatched { tag: *tag })?;
        for d in 0..3 {
            if !na[d].add(&nb[d]).is_zero() {
                return Err(Rejection::InterfaceNormalNotOpposite { tag: *tag });
            }
        }
        if ga != gb || ha != hb {
            return Err(Rejection::InterfaceFluxMismatch { tag: *tag });
        }
    }
    let total_l = report_a.lower_bound_l.add(&report_b.lower_bound_l);
    let total_c = total_l.mul(&BigRat::from_i64(-2, 1));
    Ok(ComposedProofReport {
        matched_interface_faces: if_a.len(),
        report_a,
        report_b,
        total_lower_bound_l: total_l,
        total_compliance_upper: total_c,
    })
}
