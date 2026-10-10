export AbstractLossStructure
export NotHandled
export NOT_HANDLED
export loss_structure_value
export loss_structure_gradient!
export loss_structure_hvp!

# ----------------------------------------------------------------------------- #
#        AbstractLossStructure — declared per-knot loss structure                #
# ----------------------------------------------------------------------------- #

"""
    abstract type AbstractLossStructure

Capability vocabulary for a **declared per-knot loss structure**: a small
data payload plus a rule name from which the loss's *value*, *gradient*
and *Hessian-vector product* all follow together.

This generalizes the curvature-only capability that shipped as
[`KnotHVP`](@ref) — which is now an **alias** for this type, so every
existing subtype relation, `isa` test and dispatch site holds unchanged
and the change stays additive within the current minor series.

As with its predecessor, DirectTrajOpt defines **only** the vocabulary.
No concrete structure, no rule table and no apply-math lives here: a rule
name is interpreted by the consumer (Piccolissimo), which owns the
taxonomy, the factor derivation, and every apply — including whatever
batching, device residency and allocation bounds that consumer requires.

## The three verbs, declared together

    loss_structure_value(structure, z_k, params_k)           -> Real    | NOT_HANDLED
    loss_structure_gradient!(g_k, structure, z_k, params_k)  -> nothing | NOT_HANDLED
    loss_structure_hvp!(Hv_k, structure, z_k, v_k, params_k) -> nothing | NOT_HANDLED

A rule that defines curvature alone is incomplete by construction. Value
is *required*, not optional: a consumer confirms an attachment by checking
that the structure predicts the loss value, and a structure that cannot
predict a value makes that confirmation vacuous. Gradient is required
because it is the verb evaluated once per NLP iteration for the whole
solve — the one an opaque loss pays automatic differentiation for.

Both in-place verbs **accumulate** into their output (`g .+= ∇ℓ`,
`Hv .+= ∇²ℓ·v`) so they compose with other per-knot contributions summed
into the same buffer.

## Not-handled is the default, and the default is inert

Every generic fallback returns [`NOT_HANDLED`](@ref NotHandled). With no
consumer loaded, every objective takes exactly the path it takes today.
In particular [`ConstantLowRankHVP`](@ref) and [`CustomKnotHVP`](@ref)
inherit the fallback and keep their automatic-differentiation gradient
unchanged, by design; a bespoke loss that wants the fast gradient declares
a structure instead of extending those carriers.

## The stencil axis

`z_k` is the knot block gathered over the structure's declared stencil.
The axis is part of the contract so inter-knot (width > 1) losses have a
place to land, but width 1 is the only supported width; a consumer handed
anything else must raise a thrown error rather than silently truncate.
Concrete structures carry their stencil offsets in the payload.
"""
abstract type AbstractLossStructure end

"""
    struct NotHandled end
    const NOT_HANDLED = NotHandled()

The **not-handled sentinel** returned by every
[`loss_structure_value`](@ref), [`loss_structure_gradient!`](@ref) and
[`loss_structure_hvp!`](@ref) fallback.

It is a distinct singleton *type*, deliberately not `nothing`, not a
`Symbol` and not `missing`. The in-place gradient verb returns `nothing`
when it *succeeds*, so a `nothing`-means-unhandled convention would make a
working declared gradient indistinguishable from an absent one — and the
failure mode is a silent, permanent fallback to automatic differentiation
that no test detects except by timing.

Callers branch on `ret === NOT_HANDLED`, never on `ret === nothing`.
Distinguishing the two is the caller's obligation and the reason the
sentinel exists.
"""
struct NotHandled end

const NOT_HANDLED = NotHandled()

"""
    loss_structure_value(structure, z_k, params_k) -> Real | NOT_HANDLED

Predict the per-knot loss value `ℓ(z_k, params_k)` from the declared
`structure` alone, **without** evaluating the objective's closure.

Returns [`NOT_HANDLED`](@ref NotHandled) unless a consumer has defined a
method for this structure. Value prediction is the only agreement a
consumer may check during a solve — no derivative oracle runs on the hot
path — so it is what makes runtime confirmation of a recognized structure
non-vacuous.
"""
loss_structure_value(::AbstractLossStructure, ::Any, ::Any) = NOT_HANDLED
loss_structure_value(::Nothing, ::Any, ::Any) = NOT_HANDLED

"""
    loss_structure_gradient!(g_k, structure, z_k, params_k) -> nothing | NOT_HANDLED

Accumulate `∇ℓ(z_k, params_k)` into `g_k` from the declared `structure`.

Returns `nothing` on success and [`NOT_HANDLED`](@ref NotHandled) when the
structure does not handle this call — never `nothing` for "not handled",
which is exactly the confusion the sentinel exists to prevent.
"""
loss_structure_gradient!(::Any, ::AbstractLossStructure, ::Any, ::Any) = NOT_HANDLED
loss_structure_gradient!(::Any, ::Nothing, ::Any, ::Any) = NOT_HANDLED

"""
    loss_structure_hvp!(Hv_k, structure, z_k, v_k, params_k) -> nothing | NOT_HANDLED

Accumulate the Hessian-vector product `∇²ℓ(z_k, params_k)·v_k` into `Hv_k`
from the declared `structure`, matrix-free — the direction is contracted
first and the per-knot block is never formed.

Returns `nothing` on success and [`NOT_HANDLED`](@ref NotHandled)
otherwise.
"""
loss_structure_hvp!(::Any, ::AbstractLossStructure, ::Any, ::Any, ::Any) = NOT_HANDLED
loss_structure_hvp!(::Any, ::Nothing, ::Any, ::Any, ::Any) = NOT_HANDLED

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "AbstractLossStructure — the three verbs fall back to the sentinel" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    # The sentinel is a distinct TYPE, not `nothing`, not a Symbol, not `missing`.
    # An in-place verb that SUCCEEDS returns `nothing`, so the two must never be
    # conflated (AC2).
    @test NOT_HANDLED isa NotHandled
    @test NOT_HANDLED !== nothing
    @test !(nothing isa NotHandled)
    @test !(NOT_HANDLED isa Symbol)
    @test !(NOT_HANDLED isa Missing)
    @test isbitstype(NotHandled)

    # `KnotHVP` is an ALIAS for the new abstract type, so every existing subtype
    # relation holds unchanged (AC1).
    @test KnotHVP === AbstractLossStructure
    @test ConstantLowRankHVP <: AbstractLossStructure
    @test CustomKnotHVP <: AbstractLossStructure

    z = randn(4)
    v = randn(4)
    g = zeros(4)
    Hv = zeros(4)

    # No carrier at all.
    @test loss_structure_value(nothing, z, nothing) === NOT_HANDLED
    @test loss_structure_gradient!(g, nothing, z, nothing) === NOT_HANDLED
    @test loss_structure_hvp!(Hv, nothing, z, v, nothing) === NOT_HANDLED

    # The declared rank-2 carrier and the escape hatch INHERIT the fallback and
    # keep today's behavior by design (AC6).
    clrh = ConstantLowRankHVP(randn(2, 4), :neg2_sign)
    custom = CustomKnotHVP((Hv_k, z_k, v_k, p) -> nothing, false)
    for cap in (clrh, custom)
        @test loss_structure_value(cap, z, nothing) === NOT_HANDLED
        @test loss_structure_gradient!(g, cap, z, nothing) === NOT_HANDLED
        @test loss_structure_hvp!(Hv, cap, z, v, nothing) === NOT_HANDLED
    end

    # Nothing was written by a declined call.
    @test all(iszero, g)
    @test all(iszero, Hv)
end

@testitem "AbstractLossStructure — declared structure rides the existing knot_hvp field" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt
    using DirectTrajOpt.Objectives
    using LinearAlgebra

    _, traj = bilinear_dynamics_and_trajectory()

    # A minimal consumer-side structure: ℓ(z) = c·‖z‖², declared not probed.
    struct _ScaledSqNorm <: AbstractLossStructure
        c::Float64
    end
    DirectTrajOpt.Objectives.loss_structure_value(s::_ScaledSqNorm, z, _) =
        s.c * sum(abs2, z)
    function DirectTrajOpt.Objectives.loss_structure_gradient!(g, s::_ScaledSqNorm, z, _)
        g .+= (2 * s.c) .* z
        return nothing
    end
    function DirectTrajOpt.Objectives.loss_structure_hvp!(Hv, s::_ScaledSqNorm, z, v, _)
        Hv .+= (2 * s.c) .* v
        return nothing
    end

    s = _ScaledSqNorm(3.0)
    @test s isa KnotHVP          # the alias keeps the existing vocabulary usable

    # It rides the EXISTING field and is read back through the EXISTING trait —
    # nothing renamed, nothing widened.
    obj = TerminalObjective(x -> 3.0 * norm(x)^2, :u, traj; Q = 1.0, knot_hvp = s)
    @test knot_hvp(obj, traj) === s

    z = randn(length(traj.components[:u]))
    v = randn(length(z))
    @test loss_structure_value(s, z, nothing) ≈ 3.0 * sum(abs2, z)

    g = zeros(length(z))
    @test loss_structure_gradient!(g, s, z, nothing) === nothing
    @test g ≈ 6.0 .* z

    Hv = zeros(length(z))
    @test loss_structure_hvp!(Hv, s, z, v, nothing) === nothing
    @test Hv ≈ 6.0 .* v

    # Accumulate, not overwrite — the verbs compose into a shared buffer.
    loss_structure_gradient!(g, s, z, nothing)
    @test g ≈ 12.0 .* z
end

@testitem "AbstractLossStructure — gradient! takes the declared branch, else AD" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt
    using DirectTrajOpt.Objectives
    using TrajectoryIndexingUtils
    using LinearAlgebra

    _, traj = bilinear_dynamics_and_trajectory()

    # A structure declaring a DELIBERATELY WRONG gradient, so the test can tell
    # which branch ran (AC4).
    struct _WrongGradient <: AbstractLossStructure end
    function DirectTrajOpt.Objectives.loss_structure_gradient!(g, ::_WrongGradient, z, _)
        g .+= 1.0
        return nothing
    end

    d = length(traj.components[:u])
    Q = 2.5
    Z_dim = traj.dim * traj.N + traj.global_dim
    knot_indices = slice(traj.N, traj.components[:u], traj.dim)

    # Declared ⇒ the declared branch runs (and the Qs scaling still applies).
    obj_declared =
        TerminalObjective(x -> norm(x)^2, :u, traj; Q = Q, knot_hvp = _WrongGradient())
    ∇ = zeros(Z_dim)
    gradient!(∇, obj_declared, traj)
    @test ∇[knot_indices] ≈ fill(Q, d)
    @test all(iszero, ∇[setdiff(1:Z_dim, knot_indices)])

    # Not declared ⇒ AD, unchanged.
    obj_ad = TerminalObjective(x -> norm(x)^2, :u, traj; Q = Q)
    ∇ad = zeros(Z_dim)
    gradient!(∇ad, obj_ad, traj)
    @test ∇ad[knot_indices] ≈ Q .* (2 .* traj[traj.N][:u])

    # A carrier that DECLINES (returns the sentinel) also falls to AD, unchanged —
    # the existing carriers keep their AD gradient by design (AC6).
    obj_declined = TerminalObjective(
        x -> norm(x)^2,
        :u,
        traj;
        Q = Q,
        knot_hvp = ConstantLowRankHVP(randn(2, d), :neg2_sign),
    )
    ∇decl = zeros(Z_dim)
    gradient!(∇decl, obj_declined, traj)
    @test ∇decl ≈ ∇ad
end
