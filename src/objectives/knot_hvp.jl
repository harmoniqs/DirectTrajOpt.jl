export KnotHVP
export ConstantLowRankHVP
export CustomKnotHVP
export knot_hvp

# ----------------------------------------------------------------------------- #
#                     KnotHVP — declarable per-knot HVP capability              #
# ----------------------------------------------------------------------------- #

"""
    abstract type KnotHVP

Capability carrier for **declarable, matrix-free per-knot
Hessian-vector products** on `KnotPointObjective` and other objectives.

This is the matrix-free sibling of [`get_full_hessian`](@ref). An
objective that knows its per-knot Hessian's structure (a constant
low-rank factor, a quadratic regularizer's constant Hessian, a custom
matrix-free action, …) can attach a `KnotHVP` value to its
`knot_hvp` field; downstream solvers that *consume* this capability
(e.g. Piccolissimo's Altissimo backend) then dispatch on the declared
type instead of rediscovering structure via the dense
`get_full_hessian` path or numerical probing.

DirectTrajOpt **defines only the carriers and the trait** —
[`knot_hvp`](@ref). No apply-math lives here; the *application* of
`A`, `apply!`, and the `core` rule is the consumer's concern.

Two concrete subtypes are provided:

  * [`ConstantLowRankHVP`](@ref) — declarative, framework-optimized for
    objectives of the form `ℓ ∘ (linear functional of z)` whose per-knot
    Hessian factors as `Aᵀ G A` with constant `A` and a small
    consumer-side rule `G` (e.g. the sign of the kink in coherent
    fidelity).
  * [`CustomKnotHVP`](@ref) — escape hatch for any loss; the carrier
    is just a closure plus a device-safety advertisement.

Default behavior (and only behavior, in DTO) is the no-op trait
[`knot_hvp(::AbstractObjective, ::NamedTrajectory) = nothing`]. A
consumer that sees `nothing` must fall back to its existing path
(`get_full_hessian` for the standard CPU sparse pipeline, or whatever
matrix-free fallback the consumer chooses).
"""
abstract type KnotHVP end

"""
    ConstantLowRankHVP(A::Matrix{Float64}, core::Symbol) <: KnotHVP

Declarative carrier for objectives whose per-knot Hessian factors as
`H = Aᵀ G A` with a **constant** factor `A` and a consumer-side
link-Hessian rule `core`.

The intended usage shape — entirely a consumer convention, not enforced
here — is that the consumer computes the rank-r action

    Hv ≈ Aᵀ · G(F = ‖A·x_k‖²) · (A·v)

once `A` has been uploaded to device. The carrier itself stores only
the inputs.

# Fields
- `A::Matrix{Float64}`: constant `k × m` factor; rows are the (linearly
  independent) directions that span the per-knot Hessian's range. The
  caller is responsible for scaling `A` so that the link argument
  `F = ‖A·x_k‖²` matches the consumer's expected normalization (for
  example: ket fidelity uses unit scale; unitary fidelity uses `1/n`).
- `core::Symbol`: name of the link-Hessian rule the consumer should
  apply. Established symbol so far: `:neg2_sign` (used for
  `ℓ = |1 − |S|²|` losses, with `G = −2·sign(1−F)·I`). Additional
  symbols are added as the consumer learns new shapes.

# Notes
- DTO carries **no apply-math** for `core`. The consumer (Piccolissimo
  Issue #179) interprets it.
- `A` is `Matrix{Float64}` by design — `Float64` for solver-precision
  parity and dense-`Matrix` because `A` is typically `k × m` with small
  `k` (rank), so the storage saving from a sparse representation is
  outweighed by the per-knot upload simplicity.
"""
struct ConstantLowRankHVP <: KnotHVP
    A::Matrix{Float64}
    core::Symbol
end

"""
    CustomKnotHVP(apply!::Function, on_device::Bool) <: KnotHVP

Escape-hatch carrier for objectives whose matrix-free per-knot HVP
does **not** fit the `Aᵀ G A` shape but the user (or constructor)
nonetheless has a closure that can apply it.

# Fields
- `apply!::Function`: in-place per-knot HVP action with signature

      apply!(Hv_k::AbstractVector, z_k::AbstractVector,
             v_k::AbstractVector, params_k) -> nothing

  where `Hv_k` accumulates the contribution `H_k · v_k` for the per-knot
  Hessian block at the consumer's knot index, `z_k` is the gathered
  current iterate at that knot, `v_k` is the gathered tangent direction,
  and `params_k` is the per-knot parameter slot (matching the
  `KnotPointObjective.params[k]` entry).
- `on_device::Bool`: capability advertisement.
  - `true`  ⇒ `apply!` is safe to call on device arrays (`CuArray`,
    `JLArray`, …) without `CUDA.allowscalar`-style scalar indexing; the
    consumer may call it directly on a device-resident `z_k`.
  - `false` ⇒ `apply!` is host-only; the consumer must gather the
    necessary slice to a host `Array{Float64}`, call `apply!`, and
    scatter the result back.

# Notes
- The closure is responsible for its own correctness; DTO does not
  finite-difference-validate it.
- The closure should **accumulate** into `Hv_k` (not overwrite) so that
  it composes with other per-knot contributions the consumer may sum
  in the same buffer.
"""
struct CustomKnotHVP <: KnotHVP
    apply!::Function
    on_device::Bool
end

# ----------------------------------------------------------------------------- #
#                     knot_hvp trait                                            #
# ----------------------------------------------------------------------------- #

"""
    knot_hvp(obj::AbstractObjective, traj::NamedTrajectory) -> Union{Nothing, KnotHVP}

Read the declared per-knot HVP capability for `obj` against `traj`.

The generic default returns `nothing` — every existing DTO objective
type leaves this unchanged. An objective that wants to advertise a
matrix-free per-knot HVP overrides this method (typically by storing a
`KnotHVP` instance in a `knot_hvp` field and returning it from the
trait).

The `traj` argument is part of the contract so that future objectives
can specialize on the trajectory's structure (e.g. return different
factors for free-time vs fixed-time), even though no current carrier
needs it.

Returning `nothing` is the consumer's signal to fall back to the
dense `get_full_hessian` path (or whatever fallback the consumer
chooses); see Piccolissimo Issue #179 for the consumer side.
"""
knot_hvp(::AbstractObjective, ::NamedTrajectory) = nothing

# A `knot_hvp(obj::KnotPointObjective, ::NamedTrajectory) = obj.knot_hvp`
# specialization lives in `knot_point_objectives.jl` so that the field
# lookup is co-located with the field definition.

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "KnotHVP — trait defaults to nothing for every objective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory(add_global = true)

    # KnotPointObjective (untouched field default)
    kpo = KnotPointObjective(x -> norm(x)^2, :x, traj)
    @test knot_hvp(kpo, traj) === nothing

    # QuadraticRegularizer
    quadreg = QuadraticRegularizer(:u, traj, 1.0)
    @test knot_hvp(quadreg, traj) === nothing

    # MinimumTimeObjective
    mt = MinimumTimeObjective(traj)
    @test knot_hvp(mt, traj) === nothing

    # GlobalObjective
    gobj = GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)
    @test knot_hvp(gobj, traj) === nothing

    # CompositeObjective
    composite = kpo + 0.5 * quadreg
    @test knot_hvp(composite, traj) === nothing

    # NullObjective
    @test knot_hvp(NullObjective(), traj) === nothing
end

@testitem "KnotHVP — ConstantLowRankHVP round-trips via KnotPointObjective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    A = randn(2, 4)
    rule = :neg2_sign
    cap = ConstantLowRankHVP(A, rule)

    obj = KnotPointObjective(x -> norm(x)^2, :x, traj; knot_hvp = cap)

    got = knot_hvp(obj, traj)
    @test got isa ConstantLowRankHVP
    @test got.A === A          # identity preserved (no copy)
    @test got.core === rule
end

@testitem "KnotHVP — CustomKnotHVP round-trips via KnotPointObjective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    counter = Ref(0)
    apply! = (Hv, z, v, p) -> (counter[] += 1; nothing)
    cap = CustomKnotHVP(apply!, true)

    obj = KnotPointObjective(x -> norm(x)^2, :x, traj; knot_hvp = cap)

    got = knot_hvp(obj, traj)
    @test got isa CustomKnotHVP
    @test got.on_device === true
    @test got.apply! === apply!
    # Sanity: the closure remains callable and mutates its closed-over state.
    got.apply!(Float64[], Float64[], Float64[], nothing)
    @test counter[] == 1
end

@testitem "KnotHVP — TerminalObjective threads knot_hvp keyword" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    cap = ConstantLowRankHVP(randn(2, 4), :neg2_sign)

    # Single-name TerminalObjective
    tobj_single = TerminalObjective(x -> norm(x)^2, :x, traj; knot_hvp = cap)
    @test knot_hvp(tobj_single, traj) === cap

    # Multi-name TerminalObjective
    tobj_multi = TerminalObjective(xu -> sum(xu), [:x, :u], traj; knot_hvp = cap)
    @test knot_hvp(tobj_multi, traj) === cap
end

@testitem "KnotHVP — default field value is nothing (no-regression smoke)" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    # No knot_hvp keyword — field defaults to nothing.
    obj1 = KnotPointObjective(x -> norm(x)^2, :x, traj)
    @test obj1.knot_hvp === nothing
    @test knot_hvp(obj1, traj) === nothing

    # With explicit nothing — equivalent behavior.
    obj2 = KnotPointObjective(x -> norm(x)^2, :x, traj; knot_hvp = nothing)
    @test obj2.knot_hvp === nothing
    @test knot_hvp(obj2, traj) === nothing

    # The struct still constructs through every existing outer constructor.
    obj3 = KnotPointObjective(x -> norm(x)^2, [:x], traj)           # vector-of-names
    obj4 = KnotPointObjective((x, p) -> norm(x)^2 + p, :x, traj, [1.0 for _ = 1:traj.N])
    obj5 = TerminalObjective(x -> norm(x)^2, :x, traj)
    @test obj3.knot_hvp === nothing
    @test obj4.knot_hvp === nothing
    @test obj5.knot_hvp === nothing
end
