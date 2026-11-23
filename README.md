# `nooterra-kernel`

**Zero-trust `#![no_std]`, `#![forbid(unsafe_code)]`, zero-dependency proof kernel for `.ntcert` hardware certificates.**

`nooterra-kernel` is the open, independently auditable Trusted Computing Base (TCB) of [Nooterra](https://nooterra.ai). It verifies physical, geometric, manufacturing, tolerance, and COTS interface certificates produced by the Nooterra Compiler—without trusting any solver, CAD kernel, optimizer, or neural network that proposed them.

---

## Core Guarantees

- **`#![no_std]` + `#![forbid(unsafe_code)]`** across 100% of the kernel crate.
- **Zero external dependencies** (no third-party crates, no C bindings, no `libm`).
- **Checked arithmetic in release builds** (`overflow-checks = true` in `[profile.release]`).
- **Two-Tier Sound Verification**:
  - Pure exact arbitrary-precision integer/rational verification (`BigUint`/`BigInt`/`BigRat`) for continuum conservation laws, circular friction cones, pointwise Bernstein von Mises yield, thermal `RT_0` flux, Maxwell `ND_0` Ampère conservation, curved B-Rep sandwiching, DFM/DFA, and SDT tolerance stackups.
  - Rigorous Two-Tier PSD checking for matrix cones and buckling LMIs: Rump (2006) / Higham Thm 10.14 verified shifted-Cholesky fast path with automatic exact `BigInt` fraction-free Bareiss `LDL^T` fallback.
- **Strict Tiered Proof Honesty**: Zero-discretization-error continuum proofs (`div(σ) ≡ 0`, `div(q) ≡ Q`, `curl(H) ≡ J`, pointwise yield, 3D circular friction) are explicitly distinguished from discrete FEM-subspace buckling LMIs (`K_3D(E_min) + Λ K_G(σ*) ≽ 0`).

---

## Module Architecture

| Module | Domain | Exact Verification Rule |
|---|---|---|
| [`bigint.rs`](src/bigint.rs) | Arbitrary Precision | Base-$2^{32}$ `BigUint`/`BigInt`/`BigRat`, Knuth D division, exact top-53-bit `f64` conversion |
| [`sha256.rs`](src/sha256.rs) | Cryptographic Binding | Pure `no_std` SHA-256 binding specs, STEP CAD digests, and `.ntcert` wire integrity |
| [`psd.rs`](src/psd.rs) | Matrix Cones | Two-Tier Sound PSD checker (Rump verified shifted Cholesky + exact `BigInt` Bareiss `LDL^T`) |
| [`cert.rs`](src/cert.rs), [`check.rs`](src/check.rs) | 3D Simplicial Statics | `10×10` strain-Riccati tetrahedral certificates, polytopic uncertainty, compositional contracts |
| [`hmz.rs`](src/hmz.rs) | 3D Mechanics & Contact | Locking-free `HMZ1` $H(\mathrm{div}; \mathbb{S})$ pure-integer statics, 3D circular Coulomb friction cones, exact pointwise Bernstein 3D von Mises yield |
| [`buckling.rs`](src/buckling.rs) | Buckling & Slip | Exact-integer 3D Q1 buckling LMI assembly & PSD check; zero-trust Farkas 3D circular-cone slip impossibility proofs |
| [`thermal.rs`](src/thermal.rs) | 3D Thermal CHT | Pure-integer `RT_0` heterogeneous conduction, Robin cooling jackets, Q1 primal hot-spot temperature brackets, 8-corner GD&T boxes |
| [`maxwell.rs`](src/maxwell.rs) | 3D Electromagnetics | Exact 3D discrete De Rham (`ND_0 → RT_0`) heterogeneous magnetostatics, coupled Joule-to-thermal heat bounds, reluctance torque brackets |
| [`geometry.rs`](src/geometry.rs) | CAD, DFM, DFA, GD&T | Exact rational curved B-Rep geometric sandwiching, wall/overhang/coolant powder-escape DFM, collision-free rigid-body disassembly, 3D SDT tolerance stackup |
| [`assembly.rs`](src/assembly.rs) | Unified `.ntcert` Codec | Deterministic binary wire format (`NTCT` v3) + 8-gate cross-discipline assembly verifier binding STEP CAD, tolerances, COTS ratings, and multiphysics |
| [`crates/nooterra-wasm`](crates/nooterra-wasm) | Browser / WASM Adapter | Zero-dependency `wasm32-unknown-unknown` C-ABI wrapper keeping all FFI outside the `#![forbid(unsafe_code)]` kernel |

---

## Verification & Build

```bash
# Run full zero-trust test suite
cargo test --workspace

# Compile the zero-dependency WebAssembly verifier
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p nooterra-wasm
```

---

## License

Apache-2.0 — Copyright 2026 Nooterra Systems (`https://nooterra.ai`).
