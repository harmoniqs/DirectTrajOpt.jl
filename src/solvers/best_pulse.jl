export BestPulseCallback
export CompositeIntermediateCallback
export best_trajectory, best_objective, best_iteration, best_primal_infeasibility


"""
    BestPulseCallback(prob; inf_pr_threshold=1e-4) <: AbstractIntermediateCallback

Default trajectory-tracking callback: records the iterate with the lowest
objective value seen so far whose primal infeasibility is at or below
`inf_pr_threshold`. The full primal vector is reconstructed into a
`NamedTrajectory` snapshot (`deepcopy` of `prob.trajectory` with the
iterate's data applied), accessible via `best_trajectory(cb)`.

Solver-agnostic: works under both Ipopt and MadNLP. The callback returns
`true` so it never short-circuits the solve.
"""
mutable struct BestPulseCallback <: AbstractIntermediateCallback
    problem::Any  # DirectTrajOptProblem; avoid Problems import cycle
    inf_pr_threshold::Float64
    best_obj::Float64
    best_inf_pr::Float64
    best_iter::Int
    best_primal::Vector{Float64}
    best_trajectory::Any  # ::Union{Nothing, NamedTrajectory}

    function BestPulseCallback(problem; inf_pr_threshold::Real = 1e-4)
        return new(
            problem,
            Float64(inf_pr_threshold),
            Inf,
            Inf,
            -1,
            Float64[],
            nothing,
        )
    end
end

best_trajectory(cb::BestPulseCallback) = cb.best_trajectory
best_objective(cb::BestPulseCallback) = cb.best_obj
best_iteration(cb::BestPulseCallback) = cb.best_iter
best_primal_infeasibility(cb::BestPulseCallback) = cb.best_inf_pr

function (cb::BestPulseCallback)(
    primal::AbstractVector,
    iter::Integer;
    obj_value::Float64 = NaN,
    inf_pr::Float64 = Inf,
    kwargs...,
)
    if isfinite(obj_value) && inf_pr <= cb.inf_pr_threshold && obj_value < cb.best_obj
        cb.best_obj = obj_value
        cb.best_inf_pr = inf_pr
        cb.best_iter = Int(iter)
        cb.best_primal = collect(primal)
        snapshot = deepcopy(cb.problem.trajectory)
        try
            NamedTrajectories.update!(snapshot, cb.best_primal, type = :both)
        catch err
            @warn "BestPulseCallback: failed to materialize trajectory snapshot" exception =
                err
            snapshot = nothing
        end
        cb.best_trajectory = snapshot
    end
    return true
end


"""
    CompositeIntermediateCallback(callbacks::Vector{<:AbstractIntermediateCallback})

Fan-out wrapper that invokes each child callback every iteration with the
same arguments. The composite returns `true` iff every child returned
`true`; one child returning `false` requests the solver to stop. All
children are still invoked on a stopping iteration (no short-circuit).
"""
struct CompositeIntermediateCallback <: AbstractIntermediateCallback
    callbacks::Vector{AbstractIntermediateCallback}
end

function (cb::CompositeIntermediateCallback)(
    primal::AbstractVector,
    iter::Integer;
    kwargs...,
)
    cont = true
    for c in cb.callbacks
        cont = c(primal, iter; kwargs...) && cont
    end
    return cont
end


# Coverage targets: src/solvers/best_pulse.jl

@testitem "BestPulseCallback records improvements under inf_pr threshold" setup=[
    DTOTestHelpers,
] begin
    prob, _ = make_standard_prob()
    cb = BestPulseCallback(prob; inf_pr_threshold = 1.0)

    # Feasible-enough point: bigger objective seen first.
    keep1 = cb(prob.trajectory.datavec; obj_value = 10.0, inf_pr = 0.1)
    @test keep1 == true
    @test best_objective(cb) == 10.0
    @test best_iteration(cb) == 0
    @test best_primal_infeasibility(cb) == 0.1
    @test best_trajectory(cb) isa NamedTrajectory

    # Better objective at later iteration.
    cb(prob.trajectory.datavec, 3; obj_value = 5.0, inf_pr = 0.05)
    @test best_objective(cb) == 5.0
    @test best_iteration(cb) == 3

    # Infeasible: ignored even though objective is smaller.
    cb(prob.trajectory.datavec, 4; obj_value = 1.0, inf_pr = 10.0)
    @test best_objective(cb) == 5.0
    @test best_iteration(cb) == 3

    # Worse objective: ignored.
    cb(prob.trajectory.datavec, 5; obj_value = 100.0, inf_pr = 0.01)
    @test best_objective(cb) == 5.0
end

@testitem "CompositeIntermediateCallback fans out and AND-folds returns" setup=[
    DTOTestHelpers,
] begin
    mutable struct _CountingCB <: DirectTrajOpt.AbstractIntermediateCallback
        n::Base.RefValue{Int}
        result::Bool
    end
    (cb::_CountingCB)(primal, iter; kwargs...) = (cb.n[] += 1; cb.result)

    a = _CountingCB(Ref(0), true)
    b = _CountingCB(Ref(0), false)
    comp = CompositeIntermediateCallback(
        DirectTrajOpt.AbstractIntermediateCallback[a, b],
    )

    @test comp(Float64[], 0; obj_value = 1.0, inf_pr = 0.0) == false
    @test a.n[] == 1
    @test b.n[] == 1
end

@testitem "solve! returns BestPulseCallback by default and writes JLD2" setup=[
    DTOTestHelpers,
] begin
    using JLD2
    prob, _ = make_standard_prob()
    save_path = joinpath(mktempdir(), "pulse_test.jld2")

    result =
        solve!(prob; max_iter = 5, print_level = 0, verbose = false, save_path = save_path)
    @test result isa BestPulseCallback
    @test isfile(save_path)

    data = JLD2.load(save_path)
    @test haskey(data, "trajectory")
end

@testitem "solve! with track_best=false returns nothing" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    save_path = joinpath(mktempdir(), "pulse_no_track.jld2")

    result = solve!(
        prob;
        max_iter = 3,
        print_level = 0,
        verbose = false,
        track_best = false,
        save_path = save_path,
    )
    @test result === nothing
    @test isfile(save_path)
end

@testitem "solve! with save_solution=false skips JLD2 write" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    save_path = joinpath(mktempdir(), "should_not_exist.jld2")

    solve!(
        prob;
        max_iter = 3,
        print_level = 0,
        verbose = false,
        save_solution = false,
        save_path = save_path,
    )
    @test !isfile(save_path)
end
