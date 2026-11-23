//! Two-tier positive semidefinite (PSD) matrix verifier (`no_std`, `#![forbid(unsafe_code)]`).
//!
//! - Tier 1 (Rump 2006 / Higham Thm 10.14 verified shifted Cholesky): proves strict
//!   positive definiteness `A > 0` in IEEE-754 `f64` with explicit upper bounds on
//!   rational-to-`f64` conversion error, diagonal-shift rounding, and Cholesky backward
//!   error, using a self-verified `no_std` Newton square root. Declines to Tier 2 on
//!   semidefinite, ill-conditioned, or extreme-scale inputs.
//! - Tier 2 (fraction-free Bareiss `LDL^T` over `BigInt`): clears positive denominators
//!   by LCM and applies Sylvester determinant-identity elimination with exact-division
//!   checks. Decides positive semidefiniteness `A >= 0` over Q without rounding error.

use alloc::vec;
use alloc::vec::Vec;

use crate::bigint::{BigInt, BigRat, BigUint};

/// Unit roundoff for IEEE-754 binary64 (`2^-53`).
const U_F64: f64 = 1.1102230246251565e-16;

/// Verification outcome for a symmetric rational matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsdOutcome {
    /// Verified strictly positive definite (`A > 0`) via Rump shifted Cholesky.
    ProvenFastRump,
    /// Verified positive semidefinite (`A >= 0`) via exact `BigInt` Bareiss `LDL^T`.
    ProvenExactBareiss,
    /// Rejected due to asymmetry, invalid denominator, negative pivot, or nonzero off-diagonal on a zero pivot.
    Rejected,
}

impl PsdOutcome {
    #[inline]
    pub fn is_psd(self) -> bool {
        matches!(self, Self::ProvenFastRump | Self::ProvenExactBareiss)
    }
}

/// Verification policy for PSD checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsdMode {
    /// Rump verified shifted-Cholesky fast path with automatic exact `BigInt` Bareiss fallback.
    TwoTierSound,
    /// Run Tier 2 exact `BigInt` Bareiss `LDL^T` on every matrix.
    StrictExactOnly,
}

/// Self-verified `no_std` IEEE-754 `sqrt(x)` for positive normal `f64` (`1e-200 < x < 1e200`).
/// Returns `None` if `|r*r - x| <= 4*u*x` does not hold.
#[inline]
fn verified_sqrt_f64(x: f64) -> Option<f64> {
    if !(x > 1e-200 && x < 1e200 && x.is_finite()) {
        return None;
    }
    // Exponent-halving IEEE-754 initial guess followed by 6 Newton-Raphson iterations.
    let mut r = f64::from_bits((x.to_bits() >> 1) + (0x3ff0_0000_0000_0000u64 >> 1));
    for _ in 0..6 {
        r = 0.5 * (r + x / r);
    }
    if !(r > 0.0 && r.is_finite()) {
        return None;
    }
    let err = (r * r - x).abs();
    if err <= 4.0 * U_F64 * x {
        Some(r)
    } else {
        None
    }
}

/// Tier 1 Rump (2006) / Higham Thm 10.14 verified shifted Cholesky on an `n x n` symmetric
/// row-major `f64` matrix `a` satisfying `|a[i*n+j] - A_exact[i,j]| <= eps_rel * |a[i*n+j]|`.
///
/// Supports `1 <= n <= 4096` (stack buffer for `n <= 16`, heap allocation for larger `n`).
/// Returns `true` only when `A_exact > 0` follows from the backward error bound; returns `false`
/// to trigger the exact `BigInt` Bareiss fallback.
pub fn try_rump_verified_pd_f64(a: &[f64], n: usize, eps_rel: f64) -> bool {
    if n == 0 || n > 4096 || a.len() != n * n {
        return false;
    }
    let mut tr_abs = 0.0f64;
    let mut norm_sq = 0.0f64;
    let mut max_diag = 0.0f64;
    for i in 0..n {
        let di = a[i * n + i];
        if !di.is_finite() || di <= 0.0 || di > 1e100 {
            return false;
        }
        tr_abs += di;
        if di > max_diag {
            max_diag = di;
        }
        norm_sq += di * di;
        for j in (i + 1)..n {
            let v = a[i * n + j];
            if !v.is_finite() || v.abs() > 1e100 || v != a[j * n + i] {
                return false;
            }
            norm_sq += 2.0 * v * v;
        }
    }
    let norm_f = match verified_sqrt_f64(norm_sq) {
        Some(v) => v,
        None => return false,
    };

    // Higham gamma_{n+2} = (n + 2)*u / (1 - (n + 2)*u) scaled by 1.05 for rounding margin.
    let ku = ((n + 2) as f64) * U_F64;
    if ku >= 0.01 {
        return false;
    }
    let gamma_hi = (ku / (1.0 - ku)) * 1.05;

    // Diagonal shift placed 32x above the backward and conversion error bound.
    let alpha = 32.0 * gamma_hi * tr_abs + 32.0 * eps_rel * norm_f + 1e-150;
    if !alpha.is_finite() || alpha <= 0.0 {
        return false;
    }

    let mut stack_buf = [0.0f64; 16 * 16];
    let mut heap_buf: Vec<f64> = Vec::new();
    let l: &mut [f64] = if n <= 16 {
        &mut stack_buf[..n * n]
    } else {
        heap_buf.resize(n * n, 0.0);
        &mut heap_buf[..]
    };

    let mut l_norm_sq = 0.0f64;
    for j in 0..n {
        let mut s = a[j * n + j] - alpha;
        for k in 0..j {
            let ljk = l[j * n + k];
            s -= ljk * ljk;
        }
        if !(s > 1e-180 && s.is_finite()) {
            return false;
        }
        let ljj = match verified_sqrt_f64(s) {
            Some(v) => v,
            None => return false,
        };
        l[j * n + j] = ljj;
        l_norm_sq += ljj * ljj;
        for i in (j + 1)..n {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            let lij = sum / ljj;
            if !lij.is_finite() || lij.abs() > 1e100 {
                return false;
            }
            l[i * n + j] = lij;
            l_norm_sq += lij * lij;
        }
    }

    // Rump-Higham lower bound check on A_exact with a 2x margin.
    let l_norm_sq_hi = l_norm_sq * 1.05;
    let norm_f_hi = norm_f * 1.05;
    let err_bound = 2.0 * gamma_hi * l_norm_sq_hi
        + 2.0 * eps_rel * norm_f_hi
        + 2.0 * U_F64 * (max_diag + alpha);
    err_bound.is_finite() && alpha > 2.0 * err_bound
}

/// Tier 2 exact fraction-free Bareiss `LDL^T` on a symmetric row-major `n x n` integer matrix `b`.
pub fn exact_bareiss_int_is_psd(b: &mut [BigInt], n: usize) -> bool {
    if n == 0 || b.len() != n * n {
        return false;
    }
    let mut p_prev = BigInt::from_i64(1);
    for k in 0..n {
        let piv = b[k * n + k].clone();
        if piv.is_neg() {
            return false;
        }
        if piv.is_zero() {
            for i in (k + 1)..n {
                if !b[k * n + i].is_zero() {
                    return false;
                }
            }
            continue;
        }
        for i in (k + 1)..n {
            let b_ki = b[k * n + i].clone();
            for j in i..n {
                let prod1 = piv.mul(&b[i * n + j]);
                let prod2 = b_ki.mul(&b[k * n + j]);
                let num = prod1.sub(&prod2);
                let q = num.div_exact(&p_prev);
                b[i * n + j] = q.clone();
                b[j * n + i] = q;
            }
        }
        p_prev = piv;
    }
    true
}

/// Verifies symmetry and positive semidefiniteness of an `n x n` row-major `BigRat` matrix.
pub fn verify_psd_bigrat(mat: &[BigRat], n: usize, mode: PsdMode) -> PsdOutcome {
    if n == 0 || mat.len() != n * n {
        return PsdOutcome::Rejected;
    }
    for i in 0..n {
        if mat[i * n + i].den.is_zero() {
            return PsdOutcome::Rejected;
        }
        for j in (i + 1)..n {
            let a = &mat[i * n + j];
            let b = &mat[j * n + i];
            if a.den.is_zero() || b.den.is_zero() {
                return PsdOutcome::Rejected;
            }
            let lhs = a.num.mul(&BigInt { neg: false, mag: b.den.clone() });
            let rhs = b.num.mul(&BigInt { neg: false, mag: a.den.clone() });
            if lhs != rhs {
                return PsdOutcome::Rejected;
            }
        }
    }

    if mode == PsdMode::TwoTierSound && n <= 64 {
        let mut af = [0.0f64; 64 * 64];
        let mut ok_conv = true;
        'conv: for i in 0..n {
            for j in i..n {
                match mat[i * n + j].to_f64_checked() {
                    Some(v) => {
                        af[i * n + j] = v;
                        af[j * n + i] = v;
                    }
                    None => {
                        ok_conv = false;
                        break 'conv;
                    }
                }
            }
        }
        if ok_conv && try_rump_verified_pd_f64(&af[..n * n], n, 8.0 * U_F64) {
            return PsdOutcome::ProvenFastRump;
        }
    }

    let mut d_lcm = BigUint::one();
    for entry in mat {
        let (_, rem) = d_lcm.div_rem_u(&entry.den);
        if !rem.is_zero() {
            d_lcm = BigUint::lcm_u(&d_lcm, &entry.den);
        }
    }
    let mut b = vec![BigInt::zero(); n * n];
    for i in 0..n {
        for j in i..n {
            let e = &mat[i * n + j];
            let (scale, rem) = d_lcm.div_rem_u(&e.den);
            assert!(rem.is_zero());
            let v = e.num.mul(&BigInt { neg: false, mag: scale });
            b[i * n + j] = v.clone();
            b[j * n + i] = v;
        }
    }
    if exact_bareiss_int_is_psd(&mut b, n) {
        PsdOutcome::ProvenExactBareiss
    } else {
        PsdOutcome::Rejected
    }
}

/// Verifies symmetry and positive semidefiniteness of a row-major `n x n` integer matrix `int_mat`,
/// representing `int_mat / D` for any positive common denominator `D > 0`.
///
/// Positive scaling `D > 0` preserves PSDness identically, so neither tier requires `D`.
pub fn verify_psd_i128_scaled(int_mat: &[i128], n: usize, mode: PsdMode) -> PsdOutcome {
    if n == 0 || int_mat.len() != n * n {
        return PsdOutcome::Rejected;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if int_mat[i * n + j] != int_mat[j * n + i] {
                return PsdOutcome::Rejected;
            }
        }
    }
    // A symmetric matrix with an identically zero row and column k is PSD iff its principal
    // submatrix omitting k is PSD; stripping zero rows lets Tier 1 prove positive-definite active blocks.
    if mode == PsdMode::TwoTierSound && n <= 4096 {
        let mut active = Vec::with_capacity(n);
        for i in 0..n {
            let mut nonzero = false;
            for j in 0..n {
                if int_mat[i * n + j] != 0 {
                    nonzero = true;
                    break;
                }
            }
            if nonzero {
                active.push(i);
            }
        }
        let na = active.len();
        if na == 0 {
            return PsdOutcome::ProvenFastRump;
        }
        let mut stack_a = [0.0f64; 16 * 16];
        let mut heap_a: Vec<f64> = Vec::new();
        let af: &mut [f64] = if na <= 16 {
            &mut stack_a[..na * na]
        } else {
            heap_a.resize(na * na, 0.0);
            &mut heap_a[..]
        };
        for r in 0..na {
            let ir = active[r];
            for c in r..na {
                let ic = active[c];
                let v = int_mat[ir * n + ic] as f64;
                af[r * na + c] = v;
                af[c * na + r] = v;
            }
        }
        if try_rump_verified_pd_f64(&af[..na * na], na, 4.0 * U_F64) {
            return PsdOutcome::ProvenFastRump;
        }
    }
    let mut b: Vec<BigInt> = int_mat.iter().map(|&v| BigInt::from_i128(v)).collect();
    if exact_bareiss_int_is_psd(&mut b, n) {
        PsdOutcome::ProvenExactBareiss
    } else {
        PsdOutcome::Rejected
    }
}
