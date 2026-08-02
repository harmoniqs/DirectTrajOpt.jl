export QuadraticRegularizer

using TrajectoryIndexingUtils

# ----------------------------------------------------------------------------- #
# Quadratic Regularizer
# ----------------------------------------------------------------------------- #

"""
    QuadraticRegularizer <: AbstractObjective

Quadratic regularization objective for a trajectory component.

Computes:
```math
J = \\sum_{k \\in \\text{times}} \\frac{1}{2} (v_k - v_\\text{baseline})^T R (v_k - v_\\text{baseline}) \\Delta t
```

The single Δt weight makes this a Riemann sum of the time integral
``\\frac{1}{2}\\int (v - v_\\text{baseline})^T R (v - v_\\text{baseline}) dt``,
so the value is a property of the trajectory being penalised and not of the
number of knots used to discretise it.

Gradients and Hessians are computed analytically. Because the value is linear
in each Δt, ``\\partial^2 J / \\partial \\Delta t^2`` is identically zero and is
not declared in the Hessian structure.

!!! warning "Migration: the meaning of `R` changed"
    Versions up to and including v0.9.8 weighted each knot by Δt² rather than
    the single Δt documented above (issue #122). That made the penalty fall off
    as 1/N under grid refinement — 32× weaker across a 25 → 800 knot sweep for
    a fixed continuous pulse — so any hand-tuned `R` was tuned against a
    grid-dependent quantity. On a uniform grid the old *value* is reproduced
    exactly by passing `R * Δt` — but not the old `∂J/∂Δt`, so this is not a
    drop-in substitution when the timestep is a decision variable. Problems
    with tuned regularisation weights should be re-tuned.

# Fields
- `name::Symbol`: Name of the variable to regularize
- `R::Vector{Float64}`: Diagonal weight matrix
- `baseline::Matrix{Float64}`: Baseline values (column per timestep)
- `times::Vector{Int}`: Time indices where regularization is applied

# Constructor
```julia
QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Union{Real, AbstractVector{<:Real}};
    baseline::AbstractMatrix{<:Real}=zeros(traj.dims[name], traj.N),
    times::AbstractVector{Int}=1:traj.N
)
```
"""
struct QuadraticRegularizer <: AbstractObjective
    name::Symbol
    R::Vector{Float64}
    baseline::Matrix{Float64}
    times::Vector{Int}
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    baseline::AbstractMatrix{<:Real} = zeros(traj.dims[name], traj.N),
    times::AbstractVector{Int} = 1:traj.N,
)
    @assert length(R) == traj.dims[name]

    return QuadraticRegularizer(
        name,
        Vector{Float64}(R),
        Matrix{Float64}(baseline),
        Vector{Int}(times),
    )
end

function QuadraticRegularizer(name::Symbol, traj::NamedTrajectory, R::Real; kwargs...)
    return QuadraticRegularizer(name, traj, R * ones(traj.dims[name]); kwargs...)
end

function Base.show(io::IO, reg::QuadraticRegularizer)
    R_str =
        length(reg.R) <= 4 ? string(round.(reg.R, sigdigits = 4)) :
        "[$(length(reg.R))-vector]"
    n = length(reg.times)
    times_str =
        (!isempty(reg.times) && first(reg.times) == 1 && last(reg.times) == n) ? "all" :
        "$n times"
    print(io, "QuadraticRegularizer on :$(reg.name) (R = $R_str, $times_str)")
end

# Implement AbstractObjective interface

function objective_value(reg::QuadraticRegularizer, traj::NamedTrajectory)
    J = 0.0
    for t ∈ reg.times
        zₖ = traj[t]
        vₖ = zₖ[reg.name]
        Δv = vₖ - reg.baseline[:, t]
        Δt = zₖ.timestep
        # single Δt weight: the Riemann sum of ½ Δvᵀ R Δv over time
        J += 0.5 * Δt * Δv' * (reg.R .* Δv)
    end
    return J
end

function gradient!(∇::AbstractVector, reg::QuadraticRegularizer, traj::NamedTrajectory)
    v_comps = traj.components[reg.name]
    # Defensive, matching LinearRegularizer: `NamedTrajectory.timestep` is typed
    # `Symbol` in the current NamedTrajectories, so this is always true and the
    # branch folds away. It guards the `traj.components[traj.timestep]` lookup,
    # which would be a NamedTuple indexed by a Float64 if fixed timesteps ever
    # come back.
    free_time = traj.timestep isa Symbol
    Δt_comps = free_time ? traj.components[traj.timestep] : ()
    for t ∈ reg.times
        zₖ = traj[t]
        vₖ = zₖ[reg.name]
        Δvₖ = vₖ - reg.baseline[:, t]
        Δtₖ = zₖ.timestep

        # ∂J/∂v_k = Δt_k · R ⊙ Δv_k
        ∇v = Δtₖ .* (reg.R .* Δvₖ)
        v_indices = slice(t, v_comps, traj.dim)
        ∇[v_indices] .+= ∇v

        # ∂J/∂Δt_k = ½ Δv_kᵀ R Δv_k  (if the timestep is a variable)
        if free_time
            ∇Δt = 0.5 * Δvₖ' * (reg.R .* Δvₖ)
            Δt_indices = slice(t, Δt_comps, traj.dim)
            ∇[Δt_indices] .+= ∇Δt
        end
    end

    return nothing
end

function hessian_structure(reg::QuadraticRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = spzeros(Z_dim, Z_dim)

    v_comps = traj.components[reg.name]
    free_time = traj.timestep isa Symbol
    Δt_comp = free_time ? traj.components[traj.timestep][1] : 0

    for k ∈ reg.times
        v_indices = slice(k, v_comps, traj.dim)

        # ∂²J/∂v² = Δt · diag(R). R is a vector of per-component weights, so this
        # block is diagonal; declaring the full d×d block would reserve d(d-1)/2
        # structural nonzeros per knot that can never be nonzero.
        for v_index ∈ v_indices
            structure[v_index, v_index] = 1.0
        end

        # ∂²J/∂Δt∂v, only when the timestep is a decision variable.
        if free_time
            structure[v_indices, index(k, Δt_comp, traj.dim)] .= 1.0
        end

        # No ∂²J/∂Δt² entry: with the single-Δt weight the objective is linear
        # in each Δt, so that second derivative is identically zero and must
        # not be declared as a structural nonzero.
    end

    return structure
end

function get_full_hessian(reg::QuadraticRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²J = spzeros(Z_dim, Z_dim)

    v_comps = traj.components[reg.name]
    free_time = traj.timestep isa Symbol
    Δt_comp = free_time ? traj.components[traj.timestep][1] : 0

    for t ∈ reg.times
        zₖ = traj[t]
        Δt = zₖ.timestep
        v_indices = slice(t, v_comps, traj.dim)

        # ∂²J/∂v² = Δt * R. Present whether or not Δt is a decision variable —
        # a fixed timestep is still a factor in the value.
        ∂²J[v_indices, v_indices] = Δt * spdiagm(reg.R)

        # ∂²J/∂Δt∂v = R ⊙ (v - baseline), only when the timestep is a decision
        # variable. Unlike LinearRegularizer, this objective cannot return early
        # for fixed timesteps: the ∂²J/∂v² block above survives.
        if free_time
            rₖ = zₖ[reg.name] - reg.baseline[:, t]
            ∂²J[v_indices, index(t, Δt_comp, traj.dim)] = reg.R .* rₖ
        end

        # ∂²J/∂Δt² = 0 — the value is linear in Δt, so there is no
        # timestep-timestep term. Deliberately not stored (see
        # `hessian_structure`, which declares no entry for it).
    end

    return dropzeros!(∂²J)
end

# ============================================================================ #

# ----------------------------------------------------------------------------- #
# Linear Regularizer
# ----------------------------------------------------------------------------- #

"""
    LinearRegularizer <: AbstractObjective

Linear regularization objective for a trajectory component.

Computes:
```math
J = \\sum_{k \\in \\text{times}} \\sum_i R_i \\cdot v_{k,i} \\cdot \\Delta t_k
```

Used for L1 penalty via slack variables: when applied to a non-negative slack
variable `s ≥ 0` satisfying `|du| ≤ s`, minimizing `Σ R_i s_i Δt` yields
the exact L1 norm of `du`.

Gradients and Hessians are computed analytically. The Hessian has only
cross-terms ∂²J/∂v∂Δt = R_i (no diagonal).

# Fields
- `name::Symbol`: Name of the variable to regularize
- `R::Vector{Float64}`: Per-component weights
- `times::Vector{Int}`: Time indices where regularization is applied

# Constructor
```julia
LinearRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Union{Real, AbstractVector{<:Real}};
    times::AbstractVector{Int}=1:traj.N
)
```
"""
struct LinearRegularizer <: AbstractObjective
    name::Symbol
    R::Vector{Float64}
    times::Vector{Int}
end

function LinearRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    times::AbstractVector{Int} = 1:traj.N,
)
    @assert length(R) == traj.dims[name]
    return LinearRegularizer(name, Vector{Float64}(R), Vector{Int}(times))
end

function LinearRegularizer(name::Symbol, traj::NamedTrajectory, R::Real; kwargs...)
    return LinearRegularizer(name, traj, R * ones(traj.dims[name]); kwargs...)
end

function Base.show(io::IO, reg::LinearRegularizer)
    R_str =
        length(reg.R) <= 4 ? string(round.(reg.R, sigdigits = 4)) :
        "[$(length(reg.R))-vector]"
    n = length(reg.times)
    times_str =
        (!isempty(reg.times) && first(reg.times) == 1 && last(reg.times) == n) ? "all" :
        "$n times"
    print(io, "LinearRegularizer on :$(reg.name) (R = $R_str, $times_str)")
end

# Implement AbstractObjective interface

function objective_value(reg::LinearRegularizer, traj::NamedTrajectory)
    J = 0.0
    for t ∈ reg.times
        zₖ = traj[t]
        vₖ = zₖ[reg.name]
        Δt = zₖ.timestep
        J += Δt * dot(reg.R, vₖ)
    end
    return J
end

function gradient!(∇::AbstractVector, reg::LinearRegularizer, traj::NamedTrajectory)
    v_comps = traj.components[reg.name]
    for t ∈ reg.times
        zₖ = traj[t]
        Δtₖ = zₖ.timestep

        # ∂J/∂v_{k,i} = R_i · Δt_k
        v_indices = slice(t, v_comps, traj.dim)
        ∇[v_indices] .+= reg.R .* Δtₖ

        # ∂J/∂Δt_k = Σ_i R_i · v_{k,i}
        if traj.timestep isa Symbol
            Δt_comps = traj.components[traj.timestep]
            Δt_indices = slice(t, Δt_comps, traj.dim)
            ∇[Δt_indices] .+= dot(reg.R, zₖ[reg.name])
        end
    end
    return nothing
end

function hessian_structure(reg::LinearRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = spzeros(Z_dim, Z_dim)

    if !(traj.timestep isa Symbol)
        return structure
    end

    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]

    for k ∈ reg.times
        v_indices = slice(k, v_comps, traj.dim)
        Δt_index = index(k, Δt_comp, traj.dim)

        # Only cross-terms ∂²J/∂v∂Δt
        structure[v_indices, Δt_index] .= 1.0
    end

    return structure
end

function get_full_hessian(reg::LinearRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²J = spzeros(Z_dim, Z_dim)

    if !(traj.timestep isa Symbol)
        return ∂²J
    end

    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]

    for t ∈ reg.times
        v_indices = slice(t, v_comps, traj.dim)
        Δt_index = index(t, Δt_comp, traj.dim)

        # ∂²J/∂v_{k,i}∂Δt_k = R_i
        ∂²J[v_indices, Δt_index] = reg.R
    end

    return ∂²J
end

# ============================================================================ #

@testitem "testing LinearRegularizer" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    R = 1e-2
    OBJ = LinearRegularizer(:u, traj, R)

    test_objective(OBJ, traj, atol = 1e-5)
end

@testitem "testing QuadraticRegularizer" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives
    using LinearAlgebra

    _, traj = bilinear_dynamics_and_trajectory()

    R = 1.0
    OBJ = QuadraticRegularizer(:u, traj, R)

    test_objective(OBJ, traj, atol = 1e-5)

    # The value is the single-Δt Riemann sum documented in the docstring.
    # Checked at non-uniform timesteps, with a nonzero baseline and a subset
    # of times, so a per-knot Δt mix-up cannot hide behind a uniform grid.
    Δts = [0.05, 0.11, 0.17, 0.23, 0.07, 0.13]
    nu_traj = NamedTrajectory(
        (x = randn(2, 6), u = randn(3, 6), Δt = reshape(Δts, 1, 6));
        controls = (:u, :Δt),
        timestep = :Δt,
    )

    R_vec = [0.3, 1.7, 0.9]
    baseline = randn(3, 6)
    times = [1, 3, 4, 6]
    NU_OBJ = QuadraticRegularizer(:u, nu_traj, R_vec; baseline = baseline, times = times)

    Δv(t) = nu_traj.u[:, t] - baseline[:, t]
    J_expected = sum(0.5 * Δts[t] * dot(Δv(t), R_vec .* Δv(t)) for t ∈ times)
    @test objective_value(NU_OBJ, nu_traj) ≈ J_expected

    # Finite-difference parity for the gradient and the full Hessian with the
    # timestep as a decision variable — this is where the ∂Δt and ∂v∂Δt terms
    # live, and a uniform grid with a zero baseline cannot exercise them.
    test_objective(NU_OBJ, nu_traj, atol = 1e-5)
end

@testitem "QuadraticRegularizer objective is invariant to knot count" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    # One fixed continuous pulse, discretised at knot counts spanning 32×.
    # Because the objective is the Δt-weighted Riemann sum of ½ uᵀRu, its value
    # is a property of the pulse, not of the grid: refining the grid must not
    # change the penalty. Under the Δt² weighting this same sweep fell off as
    # 1/N (32× weaker at N=800 than at N=25), silently weakening the penalty
    # exactly as the degrees of freedom grew.
    T_final = 1.0
    σ = 0.2
    pulse(t) = exp(-((t - T_final / 2) / σ)^2) * sin(2π * t / T_final)
    dpulse(t) = ForwardDiff.derivative(pulse, t)

    function objective_at(N, name)
        Δt = T_final / N
        ts = ((0:(N-1)) .+ 0.5) .* Δt
        traj = NamedTrajectory(
            (
                x = zeros(1, N),
                u = reshape(pulse.(ts), 1, N),
                du = reshape(dpulse.(ts), 1, N),
                Δt = fill(Δt, 1, N),
            );
            controls = (:u, :Δt),
            timestep = :Δt,
        )
        return objective_value(QuadraticRegularizer(name, traj, 1.0), traj)
    end

    Ns = (25, 100, 400, 800)

    # Both a control and its derivative are checked: regularising `du` is as
    # common as regularising `u`, and a caller sweeping knot count needs the
    # penalty on either to be a property of the pulse rather than of the grid.
    for (name, f) ∈ ((:u, pulse), (:du, dpulse))
        Js = [objective_at(N, name) for N ∈ Ns]

        # Invariance across the sweep, to 1%
        @test all(isapprox.(Js, Js[end], rtol = 1e-2))

        # ...and the invariant value is the time integral it claims to be:
        # ½∫₀ᵀ f(t)² dt, evaluated here by an independent fine midpoint rule.
        N_ref = 20_000
        Δt_ref = T_final / N_ref
        ts_ref = ((0:(N_ref-1)) .+ 0.5) .* Δt_ref
        J_exact = 0.5 * Δt_ref * sum(f.(ts_ref) .^ 2)
        @test all(isapprox.(Js, J_exact, rtol = 1e-2))
    end
end

@testitem "QuadraticRegularizer Hessian structure declares exactly the nonzeros" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives
    using SparseArrays
    using TrajectoryIndexingUtils

    _, traj = bilinear_dynamics_and_trajectory()
    OBJ = QuadraticRegularizer(:u, traj, 1.0)

    S = DirectTrajOpt.Objectives.hessian_structure(OBJ, traj)
    H = DirectTrajOpt.Objectives.get_full_hessian(OBJ, traj)

    # Under the single-Δt weighting the objective is linear in each Δt, so
    # ∂²J/∂Δt² is identically zero — declaring it would reserve a structural
    # nonzero that can never be nonzero.
    Δt_comp = traj.components[traj.timestep][1]
    for k ∈ OBJ.times
        Δt_index = index(k, Δt_comp, traj.dim)
        @test iszero(S[Δt_index, Δt_index])
        @test iszero(H[Δt_index, Δt_index])
    end

    # The declared structure must cover every nonzero the Hessian actually
    # produces — the evaluator writes only into declared entries, so anything
    # undeclared is silently dropped from the solver's Hessian.
    rows, cols, _ = findnz(H)
    @test all(!iszero(S[i, j]) for (i, j) ∈ zip(rows, cols))

    # ...and the converse: the structure must not over-declare either. R is a
    # vector of per-component weights, so ∂²J/∂v² is diagonal; declaring the
    # dense d×d block would reserve d(d-1)/2 entries per knot that can never be
    # nonzero. Checked on a deterministic trajectory whose weights and residuals
    # are all nonzero, so every declared entry is genuinely realised — with a
    # zero weight or a zero residual component the entry would be absent from H
    # while still (legitimately) declared, and this direction would flake.
    N = 4
    exact_traj = NamedTrajectory(
        (
            x = ones(2, N),
            u = reshape(collect(1.0:(3N)), 3, N),
            Δt = fill(0.1, 1, N),
        );
        controls = (:u, :Δt),
        timestep = :Δt,
    )
    EXACT_OBJ = QuadraticRegularizer(
        :u,
        exact_traj,
        [0.3, 1.7, 0.9];
        baseline = fill(-1.0, 3, N),
    )

    S_exact = DirectTrajOpt.Objectives.hessian_structure(EXACT_OBJ, exact_traj)
    H_exact = DirectTrajOpt.Objectives.get_full_hessian(EXACT_OBJ, exact_traj)

    @test Set(zip(findnz(S_exact)[1:2]...)) == Set(zip(findnz(H_exact)[1:2]...))
end
