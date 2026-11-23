# `nooterra-kernel`

`nooterra-kernel` is the `#![no_std]`, `#![forbid(unsafe_code)]` proof verifier for `.ntcert` hardware certificates from [Nooterra](https://nooterra.ai). It independently checks physical, geometric, manufacturing, tolerance, and COTS interface certificates produced by the Nooterra Compiler without trusting the solver, CAD kernel, or optimizer that generated the witness.

## Properties

- `#![no_std]` (outside `#[cfg(test)]`) and `#![forbid(unsafe_code)]` across the entire `nooterra-kernel` crate.
- No external crate dependencies, C bindings, or `libm`.
- Release builds trap on integer overflow (`overflow-checks = true` in `[profile.release]`).
- Exact arbitrary-precision integer and rational arithmetic (`BigUint`, `BigInt`, `BigRat`) for continuum conservation laws, 3D circular Coulomb friction cones, pointwise Bernstein von Mises yield, thermal `RT_0` flux, Maxwell `ND_0` Ampère conservation, curved B-Rep sandwiching, DFM and DFA checks, and 3D SDT tolerance stackups.
- Two-tier positive semidefinite verification for matrix cones and buckling LMIs. Tier 1 runs a Rump (2006) and Higham Thm 10.14 verified shifted-Cholesky check in `f64`, and Tier 2 falls back to exact `BigInt` fraction-free Bareiss `LDL^T`.
- Explicit separation of proof domains. Continuum conservation, pointwise yield, and 3D circular friction proofs hold without discretization error, while buckling LMIs (`K_3D(E) + Lambda * K_G(sigma*) >= 0`) certify exact positive semidefiniteness of the discrete 3D Q1 finite-element tangent matrix.

## Modules

| Module | Domain | Verification rule |
|---|---|---|
| [`bigint.rs`](src/bigint.rs) | Arbitrary precision | Base-$2^{32}$ `BigUint`, `BigInt`, and `BigRat`, Knuth Algorithm D division, and bounded top-53-bit `f64` conversion |
| [`sha256.rs`](src/sha256.rs) | Cryptographic hashing | Standalone `no_std` FIPS 180-4 SHA-256 for specification hashes, STEP CAD digests, and `.ntcert` wire integrity |
| [`psd.rs`](src/psd.rs) | Matrix cones | Two-tier PSD checker combining Rump verified shifted Cholesky with exact `BigInt` Bareiss `LDL^T` fallback |
| [`cert.rs`](src/cert.rs), [`check.rs`](src/check.rs) | 3D simplicial statics | `10x10` strain-Riccati tetrahedral certificates, polytopic material uncertainty, and compositional interface contracts |
| [`hmz.rs`](src/hmz.rs) | 3D mechanics and contact | Locking-free `HMZ1` $H(\mathrm{div}; \mathbb{S})$ exact-integer statics, 3D circular Coulomb friction cones, and pointwise Bernstein 3D von Mises yield |
| [`buckling.rs`](src/buckling.rs) | Buckling and slip | Exact-integer 3D Q1 buckling LMI assembly and PSD verification, plus Farkas 3D circular-cone slip infeasibility proofs |
| [`thermal.rs`](src/thermal.rs) | 3D thermal CHT | Exact-integer `RT_0` heterogeneous conduction, Robin cooling jackets, Q1 primal hot-spot brackets, and 8-corner GD&T boxes |
| [`maxwell.rs`](src/maxwell.rs) | 3D electromagnetics | Exact 3D discrete De Rham (`ND_0 -> RT_0`) heterogeneous magnetostatics, coupled Joule-to-thermal heat bounds, and reluctance torque brackets |
| [`geometry.rs`](src/geometry.rs) | CAD, DFM, DFA, and GD&T | Exact rational curved B-Rep geometric sandwiching, wall, overhang, and coolant powder-escape DFM, collision-free rigid-body disassembly, and 3D SDT tolerance stackup |
| [`assembly.rs`](src/assembly.rs) | `.ntcert` wire format and assembly | Deterministic binary codec (`NTCT` v3) and 8-gate actuator assembly verifier binding STEP CAD, tolerances, COTS ratings, and multiphysics |
| [`crates/nooterra-wasm`](crates/nooterra-wasm) | WebAssembly adapter | Standalone `wasm32-unknown-unknown` C-ABI wrapper keeping all FFI pointer operations outside `nooterra-kernel` |

## Building and testing

Run the workspace tests:

```bash
cargo test --workspace
```

Compile the WebAssembly verifier:

```bash
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p nooterra-wasm
```

## License

Apache-2.0. Copyright 2026 Nooterra Systems (`https://nooterra.ai`).
