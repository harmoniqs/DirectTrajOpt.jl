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
    grid-dependent quantity. On a uniform grid the old value is reproduced
    exactly by passing `R * Δt`; problems with tuned regularisation weights
    should be re-tuned.

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

    warp = traj.warp
    if warp !== nothing
        # Phase 1b (DTO#149): under a warp the Δt rows are derived — the per-knot
        # ∂J/∂Δt scatter is replaced by the SINGLE chain term
        #   ∂J/∂θⱼ = Σₖ (∂J/∂Δtₖ) · ∂Δtₖ/∂θⱼ
        # accumulated at the warp-parameter columns.
        chain = WarpPlumbing.derived_row_chain(warp, traj.N)
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        dJdΔt = zeros(length(reg.times))
        for (i, t) ∈ enumerate(reg.times)
            zₖ = traj[t]
            vₖ = zₖ[reg.name]
            Δvₖ = vₖ - reg.baseline[:, t]
            Δtₖ = zₖ.timestep

            # ∂J/∂v_k = Δt_k · R ⊙ Δv_k at the packed columns
            ∇v = Δtₖ .* (reg.R .* Δvₖ)
            v_indices = WarpPlumbing.packed_slice(traj, t, v_comps)
            ∇[v_indices] .+= ∇v

            dJdΔt[i] = 0.5 * Δvₖ' * (reg.R .* Δvₖ)
        end
        for (j, col) in enumerate(warp_cols)
            ∇[col] += sum(dJdΔt[i] * chain[t, j] for (i, t) in enumerate(reg.times))
        end
        return nothing
    end

    Δt_comps = traj.components[traj.timestep]
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
        if traj.timestep isa Symbol
            ∇Δt = 0.5 * Δvₖ' * (reg.R .* Δvₖ)
            Δt_indices = slice(t, Δt_comps, traj.dim)
            ∇[Δt_indices] .+= ∇Δt
        end
    end

    return nothing
end

function hessian_structure(reg::QuadraticRegularizer, traj::NamedTrajectory)
    Z_dim = WarpPlumbing.packed_length(traj)
    structure = spzeros(Z_dim, Z_dim)

    v_comps = traj.components[reg.name]

    if traj.warp !== nothing
        # Phase 1b: v-blocks at packed positions + (v, T) cross terms from the
        # Δtₖ(T) weight. No Δt-Δt term — the value is linear in each derived Δt.
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        for k ∈ reg.times
            v_indices = WarpPlumbing.packed_slice(traj, k, v_comps)
            structure[v_indices, v_indices] .= 1.0
            structure[v_indices, warp_cols] .= 1.0
        end
        return structure
    end

    Δt_comp = traj.components[traj.timestep][1]

    for k ∈ reg.times
        v_indices = slice(k, v_comps, traj.dim)
        Δt_index = index(k, Δt_comp, traj.dim)

        # ∂²J/∂v²
        structure[v_indices, v_indices] .= 1.0

        # ∂²J/∂Δt∂v
        structure[v_indices, Δt_index] .= 1.0

        # No ∂²J/∂Δt² entry: with the single-Δt weight the objective is linear
        # in each Δt, so that second derivative is identically zero and must
        # not be declared as a structural nonzero.
    end

    return structure
end

function get_full_hessian(reg::QuadraticRegularizer, traj::NamedTrajectory)
    Z_dim = WarpPlumbing.packed_length(traj)
    ∂²J = spzeros(Z_dim, Z_dim)

    v_comps = traj.components[reg.name]

    warp = traj.warp
    if warp !== nothing
        # Phase 1b: ∂²J/∂v² = Δt·diag(R) (packed), and the chain-weighted cross
        # terms ∂²J/∂vₖ∂θⱼ = (R ⊙ rₖ)·wₖⱼ — accumulated, since several knots
        # share the warp-parameter columns.
        chain = WarpPlumbing.derived_row_chain(warp, traj.N)
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        for t ∈ reg.times
            zₖ = traj[t]
            Δt = zₖ.timestep
            v_indices = WarpPlumbing.packed_slice(traj, t, v_comps)
            rₖ = zₖ[reg.name] - reg.baseline[:, t]

            ∂²J[v_indices, v_indices] = Δt * spdiagm(reg.R)
            for (j, col) in enumerate(warp_cols)
                ∂²J[v_indices, col] .+= (reg.R .* rₖ) .* chain[t, j]
            end
        end
        return ∂²J
    end

    Δt_comp = traj.components[traj.timestep][1]

    for t ∈ reg.times
        zₖ = traj[t]
        Δt = zₖ.timestep
        v_indices = slice(t, v_comps, traj.dim)
        Δt_index = index(t, Δt_comp, traj.dim)

        rₖ = zₖ[reg.name] - reg.baseline[:, t]

        # ∂²J/∂v² = Δt * R
        ∂²J[v_indices, v_indices] = Δt * spdiagm(reg.R)

        # ∂²J/∂Δt∂v = R ⊙ (v - baseline)
        ∂²J[v_indices, Δt_index] = reg.R .* rₖ

        # ∂²J/∂Δt² = 0 — the value is linear in Δt, so there is no
        # timestep-timestep term. Deliberately not stored (see
        # `hessian_structure`, which declares no entry for it).
    end

    return ∂²J
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

    warp = traj.warp
    if warp !== nothing
        # Phase 1b (DTO#149): single ∂J/∂T chain term replaces the per-knot scatter.
        chain = WarpPlumbing.derived_row_chain(warp, traj.N)
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        dJdΔt = zeros(length(reg.times))
        for t ∈ reg.times
            zₖ = traj[t]
            Δtₖ = zₖ.timestep

            # ∂J/∂v_{k,i} = R_i · Δt_k at the packed columns
            v_indices = WarpPlumbing.packed_slice(traj, t, v_comps)
            ∇[v_indices] .+= reg.R .* Δtₖ

            # ∂J/∂Δt_k = Σ_i R_i · v_{k,i}
            dJdΔt[findfirst(==(t), reg.times)] = dot(reg.R, zₖ[reg.name])
        end
        for (j, col) in enumerate(warp_cols)
            ∇[col] += sum(dJdΔt[i] * chain[t, j] for (i, t) in enumerate(reg.times))
        end
        return nothing
    end

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
    Z_dim = WarpPlumbing.packed_length(traj)
    structure = spzeros(Z_dim, Z_dim)

    if traj.warp !== nothing
        # Phase 1b: only cross-terms ∂²J/∂v∂T at the packed columns.
        v_comps = traj.components[reg.name]
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        for k ∈ reg.times
            v_indices = WarpPlumbing.packed_slice(traj, k, v_comps)
            structure[v_indices, warp_cols] .= 1.0
        end
        return structure
    end

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
    Z_dim = WarpPlumbing.packed_length(traj)
    ∂²J = spzeros(Z_dim, Z_dim)

    warp = traj.warp
    if warp !== nothing
        # Phase 1b: cross-terms ∂²J/∂vₖ∂θⱼ = R·wₖⱼ, accumulated.
        chain = WarpPlumbing.derived_row_chain(warp, traj.N)
        v_comps = traj.components[reg.name]
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        for t ∈ reg.times
            v_indices = WarpPlumbing.packed_slice(traj, t, v_comps)
            for (j, col) in enumerate(warp_cols)
                ∂²J[v_indices, col] .+= reg.R .* chain[t, j]
            end
        end
        return ∂²J
    end

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

@testitem "QuadraticRegularizer Hessian structure declares no Δt-Δt entry" begin
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

    # The declared structure must still cover every nonzero the Hessian
    # actually produces — a solver only writes into declared entries.
    rows, cols, _ = findnz(H)
    @test all(!iszero(S[i, j]) for (i, j) ∈ zip(rows, cols))
end

# ============================================================================ #
# Phase 1b (DTO#149): derived-Δt weighting under a time warp
# ============================================================================ #

@testitem "QuadraticRegularizer under a warp: derived-Δt weight + single ∂J/∂T chain" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives
    using DirectTrajOpt.WarpPlumbing
    using NamedTrajectories
    using FiniteDiff
    using Test

    traj = warped_derivative_trajectory(N = 10, T = 1.3; seed = 7)
    R = [0.7, 1.9]
    baseline = randn(2, traj.N)
    times = [1, 3, 4, 9]
    reg = QuadraticRegularizer(:x, traj, R; baseline = baseline, times = times)

    # value: ½Σ Δtₖ(T)‖R·Δvₖ‖² with the DERIVED row as weight — unchanged reader
    Δv(t) = traj.x[:, t] - baseline[:, t]
    J_expected = sum(0.5 * traj.Δt[t] * dot(Δv(t), R .* Δv(t)) for t in times)
    @test objective_value(reg, traj) ≈ J_expected

    # gradient FD parity over the packed vector (perturbs the warp parameter T too)
    ∇ = zeros(length(traj))
    gradient!(∇, reg, traj)
    ∇_fd = FiniteDiff.finite_difference_gradient(collect(vec(traj))) do Z⃗
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        return objective_value(reg, t2)
    end
    @test all(isapprox.(∇, ∇_fd, atol = 1e-6, rtol = 1e-6))

    # the single ∂J/∂T chain term replaces the per-knot ∂J/∂Δt scatter: the only
    # nonzeros are the regularized components' columns and the T column itself
    T_col = WarpPlumbing.warp_param_indices(traj)[1]
    allowed = union(
        [WarpPlumbing.packed_slice(traj, t, traj.components[:x]) for t in times]...,
        [T_col],
    )
    @test ⊆(findall(!iszero, ∇), allowed)
    # ...and it is exact for GlobalScale: Σₜ ½ΔvᵀRΔv · wₖ
    @test ∇[T_col] ≈ sum(0.5 * dot(Δv(t), R .* Δv(t)) for t in times) * (1 / (traj.N - 1))

    # Hessian FD parity + structure covers every nonzero the Hessian produces
    H = get_full_hessian(reg, traj)
    H_fd = FiniteDiff.finite_difference_hessian(collect(vec(traj))) do Z⃗
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        return objective_value(reg, t2)
    end
    @test all(isapprox.(triu(H), triu(H_fd), atol = 1e-5, rtol = 1e-5))
    S = DirectTrajOpt.Objectives.hessian_structure(reg, traj)
    rows, cols, _ = findnz(H)
    @test all(!iszero(S[i, j]) for (i, j) in zip(rows, cols))
end

@testitem "QuadraticRegularizer under a warp is invariant to knot count" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives
    using NamedTrajectories
    using Test

    # The #122 lesson under a warp: the weight is the DERIVED Δt (a lattice
    # fraction of T), so the value is a property of the pulse, not of the grid —
    # structurally, not by careful bookkeeping.
    T = 1.0
    pulse(t) = exp(-((t - T / 2) / 0.2)^2) * sin(2π * t / T)

    function objective_at(N)
        traj = NamedTrajectory(
            (
                x = zeros(1, N),
                u = reshape(pulse.(T .* ((0:(N-1)) ./ (N - 1))), 1, N),
                Δt = fill(T / (N - 1), 1, N),
            );
            controls = (:u,),
            timestep = :Δt,
            warp = GlobalScale(T),
        )
        return objective_value(QuadraticRegularizer(:u, traj, 1.0), traj)
    end

    Ns = (25, 100, 400, 800)
    Js = [objective_at(N) for N in Ns]
    @test all(isapprox.(Js, Js[end], rtol = 1e-2))

    # and the invariant value is the time integral it claims to be
    N_ref = 20_000
    ts_ref = ((0:(N_ref-1)) .+ 0.5) .* (T / N_ref)
    J_exact = (T / N_ref) * sum(pulse.(ts_ref) .^ 2) / 2
    @test all(isapprox.(Js, J_exact, rtol = 1e-2))
end

@testitem "LinearRegularizer under a warp: derived-Δt weight + single ∂J/∂T chain" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives
    using DirectTrajOpt.WarpPlumbing
    using NamedTrajectories
    using FiniteDiff
    using Test

    traj = warped_derivative_trajectory(N = 9, T = 2.1; seed = 3)
    R = [0.4, 2.2]
    times = [2, 4, 5, 8]
    reg = LinearRegularizer(:x, traj, R; times = times)

    Δt_row(t) = traj.Δt[t]
    J_expected = sum(Δt_row(t) * dot(R, traj.x[:, t]) for t in times)
    @test objective_value(reg, traj) ≈ J_expected

    ∇ = zeros(length(traj))
    gradient!(∇, reg, traj)
    ∇_fd = FiniteDiff.finite_difference_gradient(collect(vec(traj))) do Z⃗
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        return objective_value(reg, t2)
    end
    @test all(isapprox.(∇, ∇_fd, atol = 1e-6, rtol = 1e-6))

    T_col = WarpPlumbing.warp_param_indices(traj)[1]
    @test ∇[T_col] ≈ sum(dot(R, traj.x[:, t]) for t in times) * (1 / (traj.N - 1))

    H = get_full_hessian(reg, traj)
    H_fd = FiniteDiff.finite_difference_hessian(collect(vec(traj))) do Z⃗
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        return objective_value(reg, t2)
    end
    @test all(isapprox.(triu(H), triu(H_fd), atol = 1e-5, rtol = 1e-5))
    S = DirectTrajOpt.Objectives.hessian_structure(reg, traj)
    rows, cols, _ = findnz(H)
    @test all(!iszero(S[i, j]) for (i, j) in zip(rows, cols))
end
