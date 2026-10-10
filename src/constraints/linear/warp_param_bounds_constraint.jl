export WarpParamBoundsConstraint

"""
    struct WarpParamBoundsConstraint <: AbstractLinearConstraint

Box bounds on the warp parameters — the trailing block of the packed decision vector
under a time warp (NamedTrajectories#161). For a `GlobalScale` warp this is the ONE
duration variable `T`: `lo ≤ T ≤ hi` replaces the per-knot `Δt ≥ 0` auto-bound of the
free-Δt formulation (a free duration is unbounded below without it once
`MinimumTimeObjective` reduces to `D·T`).

Refused at application when the trajectory carries no warp — there are no warp
parameters to bound. (Per-knot timestep bounds remain refused under a warp; see
`DirectTrajOptProblem`'s refusal pass, DTO#149.)

# Fields
- `lo::Vector{Float64}`: lower bounds, one per warp parameter
- `hi::Vector{Float64}`: upper bounds, one per warp parameter
- `label::String`: constraint label
"""
struct WarpParamBoundsConstraint <: AbstractLinearConstraint
    lo::Vector{Float64}
    hi::Vector{Float64}
    label::String

    function WarpParamBoundsConstraint(
        lo::AbstractVector{<:Real},
        hi::AbstractVector{<:Real};
        label = "warp parameter bounds constraint",
    )
        length(lo) == length(hi) ||
            throw(ArgumentError("lo and hi must have the same length"))
        all(l ≤ h for (l, h) in zip(lo, hi)) ||
            throw(ArgumentError("lower bounds must not exceed upper bounds"))
        return new(Vector{Float64}(lo), Vector{Float64}(hi), label)
    end
end

WarpParamBoundsConstraint(lo::Real, hi::Real; kwargs...) =
    WarpParamBoundsConstraint([lo], [hi]; kwargs...)

function Base.show(io::IO, c::WarpParamBoundsConstraint)
    print(io, "WarpParamBoundsConstraint: $(c.label) (bounds length $(length(c.lo)))")
end

# =========================================================================== #

@testitem "WarpParamBoundsConstraint application" begin
    include("../../../test/test_utils.jl")
    using NamedTrajectories
    using DirectTrajOpt: Solvers
    import MathOptInterface as MOI
    import Ipopt
    using Test

    traj = warped_derivative_trajectory(N = 6, T = 1.0)

    con = WarpParamBoundsConstraint(0.2, 5.0)
    @test sprint(show, con) isa String

    # applies GreaterThan + LessThan at the warp-parameter positions
    opt = Ipopt.Optimizer()
    vars = MOI.add_variables(opt, length(traj))
    con(opt, vars, traj)
    warp_var = vars[first(DirectTrajOpt.WarpPlumbing.warp_param_indices(traj))]
    gt = MOI.get(
        opt,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{Float64}}(),
    )
    @test any(gt) do ci
        MOI.get(opt, MOI.ConstraintFunction(), ci) == warp_var &&
            MOI.get(opt, MOI.ConstraintSet(), ci).lower == 0.2
    end
    lt =
        MOI.get(opt, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{Float64}}())
    @test any(lt) do ci
        MOI.get(opt, MOI.ConstraintFunction(), ci) == warp_var &&
            MOI.get(opt, MOI.ConstraintSet(), ci).upper == 5.0
    end

    # refused without a warp — no warp parameters to bound
    plain = NamedTrajectory(
        (x = randn(2, 4), u = randn(1, 4), Δt = fill(0.2, 1, 4));
        controls = (:u,),
        timestep = :Δt,
    )
    @test_throws ArgumentError con(opt, vars, plain)

    # constructor validation
    @test_throws ArgumentError WarpParamBoundsConstraint([0.5], [0.2])
    @test_throws ArgumentError WarpParamBoundsConstraint([0.2, 0.3], [5.0])
end
