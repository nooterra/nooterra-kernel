//! Arbitrary-precision unsigned and signed integers and exact rationals (`no_std`).
//!
//! Uses base-2^32 little-endian limbs, Knuth Algorithm D division, Euclidean GCD/LCM,
//! and top-53-bit rational-to-`f64` conversion with a bounded relative error (`<= 8 * 2^-53`)
//! without `libm`.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigUint {
    pub limbs: Vec<u32>, // little-endian, canonicalized (no trailing 0 limbs)
}

impl Ord for BigUint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_u(other)
    }
}

impl PartialOrd for BigUint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp_u(other))
    }
}

impl BigUint {
    #[inline]
    pub fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    #[inline]
    pub fn one() -> Self {
        Self { limbs: vec![1] }
    }

    pub fn from_u64(mut v: u64) -> Self {
        let mut limbs = Vec::new();
        while v > 0 {
            limbs.push(v as u32);
            v >>= 32;
        }
        Self { limbs }
    }

    pub fn from_u128(mut v: u128) -> Self {
        let mut limbs = Vec::new();
        while v > 0 {
            limbs.push(v as u32);
            v >>= 32;
        }
        Self { limbs }
    }

    #[inline]
    pub fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Exact number of significant bits (`0` has bit_len `0`).
    pub fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() - 1) * 32 + (32 - top.leading_zeros() as usize),
        }
    }

    /// Extracts the top 53 bits (or all bits if `bit_len <= 53`) as `(mantissa_u64, bit_shift)`
    /// such that `mantissa * 2^shift <= self < (mantissa + 1) * 2^shift`, with relative error <= 2^-52.
    pub fn top_53_bits(&self) -> (u64, usize) {
        let bl = self.bit_len();
        if bl == 0 {
            return (0, 0);
        }
        let shift = bl.saturating_sub(53);
        let word_idx = shift / 32;
        let bit_off = shift % 32;
        let w0 = *self.limbs.get(word_idx).unwrap_or(&0) as u128;
        let w1 = *self.limbs.get(word_idx + 1).unwrap_or(&0) as u128;
        let w2 = *self.limbs.get(word_idx + 2).unwrap_or(&0) as u128;
        let combined = w0 | (w1 << 32) | (w2 << 64);
        let mant = (combined >> bit_off) as u64;
        (mant, shift)
    }

    pub fn cmp_u(&self, other: &Self) -> Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
        }
        Ordering::Equal
    }

    pub fn add_u(&self, other: &Self) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0) as u64;
            let b = *other.limbs.get(i).unwrap_or(&0) as u64;
            let s = a + b + carry;
            out.push(s as u32);
            carry = s >> 32;
        }
        if carry > 0 {
            out.push(carry as u32);
        }
        Self { limbs: out }
    }

    /// Precondition: `self >= other`.
    pub fn sub_u(&self, other: &Self) -> Self {
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i64;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i64;
            let b = *other.limbs.get(i).unwrap_or(&0) as i64;
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1i64 << 32;
                borrow = 1;
            } else {
                borrow = 0;
            }
            out.push(d as u32);
        }
        assert_eq!(borrow, 0, "BigUint::sub_u underflow");
        let mut r = Self { limbs: out };
        r.trim();
        r
    }

    pub fn mul_u(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![0u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u64;
            let a64 = a as u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let p = a64 * (b as u64) + (out[i + j] as u64) + carry;
                out[i + j] = p as u32;
                carry = p >> 32;
            }
            out[i + other.limbs.len()] = carry as u32;
        }
        let mut r = Self { limbs: out };
        r.trim();
        r
    }

    fn shl_small(&self, bits: u32) -> Self {
        if bits == 0 || self.is_zero() {
            return self.clone();
        }
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u32;
        for &a in &self.limbs {
            out.push((a << bits) | carry);
            carry = a >> (32 - bits);
        }
        if carry > 0 {
            out.push(carry);
        }
        Self { limbs: out }
    }

    fn shr_small(&self, bits: u32) -> Self {
        if bits == 0 || self.is_zero() {
            return self.clone();
        }
        let mut out = vec![0u32; self.limbs.len()];
        let mut carry = 0u32;
        for i in (0..self.limbs.len()).rev() {
            let a = self.limbs[i];
            out[i] = (a >> bits) | carry;
            carry = a << (32 - bits);
        }
        let mut r = Self { limbs: out };
        r.trim();
        r
    }

    fn div_rem_u32(&self, d: u32) -> (Self, u32) {
        assert!(d != 0, "BigUint division by zero");
        let d64 = d as u64;
        let mut q = vec![0u32; self.limbs.len()];
        let mut rem = 0u64;
        for i in (0..self.limbs.len()).rev() {
            let cur = (rem << 32) | (self.limbs[i] as u64);
            q[i] = (cur / d64) as u32;
            rem = cur % d64;
        }
        let mut qn = Self { limbs: q };
        qn.trim();
        (qn, rem as u32)
    }

    /// Exact Knuth Algorithm D division with remainder.
    pub fn div_rem_u(&self, d: &Self) -> (Self, Self) {
        assert!(!d.is_zero(), "BigUint division by zero");
        match self.cmp_u(d) {
            Ordering::Less => return (Self::zero(), self.clone()),
            Ordering::Equal => return (Self::one(), Self::zero()),
            Ordering::Greater => {}
        }
        if d.limbs.len() == 1 {
            let (q, r) = self.div_rem_u32(d.limbs[0]);
            return (q, Self::from_u64(r as u64));
        }
        let n = d.limbs.len();
        let m = self.limbs.len() - n;
        let shift = d.limbs[n - 1].leading_zeros();
        let vn = d.shl_small(shift);
        let mut un = self.shl_small(shift).limbs;
        while un.len() <= m + n {
            un.push(0);
        }
        let v_hi = vn.limbs[n - 1] as u64;
        let v_hi2 = vn.limbs[n - 2] as u64;
        let base = 1u64 << 32;
        let mut q = vec![0u32; m + 1];

        for j in (0..=m).rev() {
            let num = ((un[j + n] as u64) << 32) | (un[j + n - 1] as u64);
            let mut qhat = num / v_hi;
            let mut rhat = num % v_hi;
            while qhat >= base || qhat * v_hi2 > (rhat << 32) + (un[j + n - 2] as u64) {
                qhat -= 1;
                rhat += v_hi;
                if rhat >= base {
                    break;
                }
            }
            let mut borrow = 0i128;
            for i in 0..n {
                let p = (qhat as i128) * (vn.limbs[i] as i128);
                let diff = (un[j + i] as i128) - (p & 0xFFFF_FFFF) - borrow;
                un[j + i] = (diff & 0xFFFF_FFFF) as u32;
                borrow = (p >> 32) - (diff >> 32);
            }
            let diff = (un[j + n] as i128) - borrow;
            un[j + n] = (diff & 0xFFFF_FFFF) as u32;
            if diff < 0 {
                qhat -= 1;
                let mut carry = 0u64;
                for i in 0..n {
                    let s = (un[j + i] as u64) + (vn.limbs[i] as u64) + carry;
                    un[j + i] = s as u32;
                    carry = s >> 32;
                }
                un[j + n] = un[j + n].wrapping_add(carry as u32);
            }
            q[j] = qhat as u32;
        }
        let mut qn = Self { limbs: q };
        qn.trim();
        let mut rem = Self { limbs: un[0..n].to_vec() };
        rem.trim();
        (qn, rem.shr_small(shift))
    }

    pub fn gcd_u(mut a: Self, mut b: Self) -> Self {
        while !b.is_zero() {
            let (_, r) = a.div_rem_u(&b);
            a = b;
            b = r;
        }
        a
    }

    pub fn lcm_u(a: &Self, b: &Self) -> Self {
        if a.is_zero() || b.is_zero() {
            return Self::zero();
        }
        let g = Self::gcd_u(a.clone(), b.clone());
        let (q, r) = a.div_rem_u(&g);
        assert!(r.is_zero());
        q.mul_u(b)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigInt {
    pub neg: bool,
    pub mag: BigUint,
}

impl BigInt {
    #[inline]
    pub fn zero() -> Self {
        Self { neg: false, mag: BigUint::zero() }
    }

    pub fn from_i64(v: i64) -> Self {
        if v == 0 {
            Self::zero()
        } else if v < 0 {
            Self { neg: true, mag: BigUint::from_u64(v.unsigned_abs()) }
        } else {
            Self { neg: false, mag: BigUint::from_u64(v as u64) }
        }
    }

    pub fn from_i128(v: i128) -> Self {
        if v == 0 {
            Self::zero()
        } else if v < 0 {
            Self { neg: true, mag: BigUint::from_u128(v.unsigned_abs()) }
        } else {
            Self { neg: false, mag: BigUint::from_u128(v as u128) }
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.mag.is_zero()
    }

    #[inline]
    pub fn is_neg(&self) -> bool {
        self.neg && !self.mag.is_zero()
    }

    #[inline]
    pub fn is_pos(&self) -> bool {
        !self.neg && !self.mag.is_zero()
    }

    pub fn negate(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            Self { neg: !self.neg, mag: self.mag.clone() }
        }
    }

    pub fn cmp_i(&self, other: &Self) -> Ordering {
        match (self.is_neg(), other.is_neg()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.mag.cmp_u(&other.mag),
            (true, true) => other.mag.cmp_u(&self.mag),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.neg == other.neg {
            let mag = self.mag.add_u(&other.mag);
            Self { neg: self.neg && !mag.is_zero(), mag }
        } else {
            match self.mag.cmp_u(&other.mag) {
                Ordering::Equal => Self::zero(),
                Ordering::Greater => Self { neg: self.neg, mag: self.mag.sub_u(&other.mag) },
                Ordering::Less => Self { neg: other.neg, mag: other.mag.sub_u(&self.mag) },
            }
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mag = self.mag.mul_u(&other.mag);
        Self { neg: (self.neg ^ other.neg) && !mag.is_zero(), mag }
    }

    /// Exact division asserting zero remainder (algebraic checksum for Bareiss LDL^T).
    pub fn div_exact(&self, divisor: &Self) -> Self {
        assert!(!divisor.is_zero(), "BigInt::div_exact by zero");
        let (q, r) = self.mag.div_rem_u(&divisor.mag);
        assert!(r.is_zero(), "Non-zero remainder in exact Bareiss division");
        Self { neg: (self.neg ^ divisor.neg) && !q.is_zero(), mag: q }
    }
}

/// Exact rational number `num / den` with `den > 0` (always reduced by GCD when constructed via `new`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigRat {
    pub num: BigInt,
    pub den: BigUint,
}

impl BigRat {
    pub fn zero() -> Self {
        Self { num: BigInt::zero(), den: BigUint::one() }
    }

    pub fn from_i64(n: i64, d: u64) -> Self {
        assert!(d > 0, "BigRat denominator must be positive");
        Self::new(BigInt::from_i64(n), BigUint::from_u64(d))
    }

    pub fn from_i128(n: i128, d: u128) -> Self {
        assert!(d > 0, "BigRat denominator must be positive");
        Self::new(BigInt::from_i128(n), BigUint::from_u128(d))
    }

    pub fn new(num: BigInt, den: BigUint) -> Self {
        assert!(!den.is_zero(), "BigRat denominator must be positive");
        if num.is_zero() {
            return Self::zero();
        }
        let g = BigUint::gcd_u(num.mag.clone(), den.clone());
        let (qn, rn) = num.mag.div_rem_u(&g);
        let (qd, rd) = den.div_rem_u(&g);
        assert!(rn.is_zero() && rd.is_zero());
        Self { num: BigInt { neg: num.neg, mag: qn }, den: qd }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    #[inline]
    pub fn is_pos(&self) -> bool {
        self.num.is_pos()
    }

    #[inline]
    pub fn is_neg(&self) -> bool {
        self.num.is_neg()
    }

    pub fn negate(&self) -> Self {
        Self { num: self.num.negate(), den: self.den.clone() }
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        if self.den == other.den {
            return Self::new(self.num.add(&other.num), self.den.clone());
        }
        let n1 = self.num.mul(&BigInt { neg: false, mag: other.den.clone() });
        let n2 = other.num.mul(&BigInt { neg: false, mag: self.den.clone() });
        Self::new(n1.add(&n2), self.den.mul_u(&other.den))
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        Self::new(self.num.mul(&other.num), self.den.mul_u(&other.den))
    }

    pub fn div(&self, other: &Self) -> Self {
        assert!(!other.is_zero(), "BigRat division by zero");
        if self.is_zero() {
            return Self::zero();
        }
        let neg = self.num.neg ^ other.num.neg;
        let num_mag = self.num.mag.mul_u(&other.den);
        let den_mag = self.den.mul_u(&other.num.mag);
        Self::new(BigInt { neg, mag: num_mag }, den_mag)
    }

    pub fn cmp_r(&self, other: &Self) -> Ordering {
        let lhs = self.num.mul(&BigInt { neg: false, mag: other.den.clone() });
        let rhs = other.num.mul(&BigInt { neg: false, mag: self.den.clone() });
        lhs.cmp_i(&rhs)
    }
}

impl PartialOrd for BigRat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigRat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_r(other)
    }
}

impl BigRat {

    /// Converts to `f64` with relative error `<= 8 * 2^-53` without `libm`.
    ///
    /// Returns `None` if the binary exponent is outside `[-300, 300]` or the magnitude exceeds
    /// `1e100`, signaling the PSD verifier to fall back to exact `BigInt` Bareiss.
    pub fn to_f64_checked(&self) -> Option<f64> {
        if self.is_zero() {
            return Some(0.0);
        }
        let (mn, sn) = self.num.mag.top_53_bits();
        let (md, sd) = self.den.top_53_bits();
        let exp = (sn as i64) - (sd as i64);
        if !(-300..=300).contains(&exp) {
            return None;
        }
        // IEEE-754 binary64 encoding of 2^exp with bias 1023.
        let pow2 = f64::from_bits(((1023i64 + exp) as u64) << 52);
        let v = ((mn as f64) / (md as f64)) * pow2;
        if !v.is_finite() || v.abs() > 1e100 {
            return None;
        }
        Some(if self.num.neg { -v } else { v })
    }
}
