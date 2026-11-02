module Problems

export DirectTrajOptProblem
export show_problem_details

export get_trajectory_constraints

using ..Objectives
using ..Integrators
using ..Constraints

using TrajectoryIndexingUtils
using NamedTrajectories
using TestItems
using LinearAlgebra

"""
    mutable struct DirectTrajOptProblem

A direct trajectory optimization problem containing all information needed for setup and solution.

# Fields
- `trajectory::NamedTrajectory`: The trajectory containing optimization variables and data
- `objective::AbstractObjective`: The objective function to minimize
- `integrators::Vector{<:AbstractIntegrator}`: The integrators defining system dynamics
- `constraints::Vector{<:AbstractConstraint}`: Constraints on the trajectory

# Constructors
```julia
DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrators::Vector{<:AbstractIntegrator};
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]
)
```

Create a problem from a trajectory, objective, and integrators. Trajectory constraints
(initial, final, bounds) are automatically extracted and added to the constraint list.
The dynamics object is created by the evaluator at solve time.

# Example
```julia
traj = NamedTrajectory((x = rand(2, 10), u = rand(1, 10)), timestep=:Δt)
obj = QuadraticRegularizer(:u, traj, 1.0)
integrator = BilinearIntegrator(G, :x, :u)
prob = DirectTrajOptProblem(traj, obj, integrator)
```
"""
mutable struct DirectTrajOptProblem
    trajectory::NamedTrajectory
    objective::AbstractObjective
    integrators::Vector{<:AbstractIntegrator}
    constraints::Vector{<:AbstractConstraint}
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrators::Vector{<:AbstractIntegrator};
    constraints::Vector{<:AbstractConstraint} = AbstractConstraint[],
)
    # Validate timestep bounds if trajectory has a timestep variable
    timestep_var = traj.timestep
    if traj.warp !== nothing
        # Phase 1b (DTO#149): under a warp the timestep rows are DERIVED — the
        # auto Δt ≥ 0 injection is structurally meaningless (it would also
        # silently drop the warp by rebuilding the trajectory without it).
        # Refuse to run it, with a pointer to the warp parameterization.
        if timestep_var isa Symbol && !haskey(traj.bounds, timestep_var)
            @info """
                Time warp present: the timestep component :$timestep_var is DERIVED \
                from the warp (NamedTrajectories#161), so the auto Δt ≥ 0 bound is \
                skipped — it is never decision data, and the warp's monotonicity \
                (validated at construction) keeps every derived Δt positive. \
                Bound the warp parameter(s) instead — e.g. \
                WarpParamBoundsConstraint(lo, hi) on the duration variable T \
                (DirectTrajOpt#149).
                """ maxlog = 1
        end
    elseif timestep_var isa Symbol && !haskey(traj.bounds, timestep_var)
        @warn """
            Trajectory has timestep variable :$timestep_var but no bounds on it.
            Adding default lower bound of 0 to prevent negative timesteps.

            Recommended: Add explicit bounds when creating the trajectory:
              NamedTrajectory(...; Δt_bounds=(min, max))
            Example:
              NamedTrajectory(qtraj, N; Δt_bounds=(1e-3, 0.5))

            Or use timesteps_all_equal=true in problem options to fix timesteps.
            """ maxlog=1

        # Add lower bound of 0 to prevent negative timesteps
        # Create new trajectory with updated bounds
        timestep_dim = traj.dims[timestep_var]
        new_bounds = merge(
            traj.bounds,
            (; timestep_var => (zeros(timestep_dim), fill(Inf, timestep_dim))),
        )

        # Extract component data
        comps_data = NamedTuple(name => traj[name] for name in traj.names)

        # Extract global component data if present
        if traj.global_dim > 0
            gcomps_data = NamedTuple(
                name => Vector(traj.global_data[traj.global_components[name]]) for
                name in keys(traj.global_components)
            )
            traj = NamedTrajectory(
                comps_data,
                gcomps_data;
                timestep = traj.timestep,
                controls = traj.control_names,
                bounds = new_bounds,
                initial = traj.initial,
                final = traj.final,
                goal = traj.goal,
            )
        else
            traj = NamedTrajectory(
                comps_data;
                timestep = traj.timestep,
                controls = traj.control_names,
                bounds = new_bounds,
                initial = traj.initial,
                final = traj.final,
                goal = traj.goal,
            )
        end
    end

    traj_constraints = get_trajectory_constraints(traj)
    # Convert to AbstractConstraint vector to allow mixed types
    all_constraints = AbstractConstraint[constraints..., traj_constraints...]
    traj.warp !== nothing &&
        _assert_warp_compatible(traj, obj, integrators, all_constraints)
    return DirectTrajOptProblem(traj, obj, integrators, all_constraints)
end

"""
    _assert_warp_compatible(traj, obj, integrators, constraints)

Phase 1b (DTO#149) refusal pass: under a time warp the timestep rows are DERIVED
(NeverTrajectories#161 — never decision data), so the per-knot time machinery of the
free-Δt formulation is structurally meaningless. Refuse at construction, with pointer
messages naming the warp-parameterization replacement.
"""
function _assert_warp_compatible(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint},
)
    traj.warp === nothing && return
    dt_name = traj.timestep
    _resolves_to_timestep(name::Symbol) = name == :Δt || name == dt_name

    for c in constraints
        if c isa TimeConsistencyConstraint
            throw(
                ArgumentError(
                    "TimeConsistencyConstraint is obsolete under a time warp: the timestep " *
                    "rows are derived from the warp (NamedTrajectories#161), so " *
                    "tₖ₊₁ = tₖ + Δtₖ holds by construction. Drop the constraint — physical " *
                    "knot times are knot_times(traj.warp, traj.N). (DirectTrajOpt#149)",
                ),
            )
        elseif c isa AllEqualConstraint && _resolves_to_timestep(c.var_name)
            throw(
                ArgumentError(
                    "TimeStepsAllEqualConstraint (AllEqualConstraint on :$(c.var_name)) is " *
                    "obsolete under a time warp: the uniform mesh is already built into the " *
                    "warp lattice, and free duration is the single warp parameter T. Remove " *
                    "the constraint and bound T instead (WarpParamBoundsConstraint). " *
                    "(DirectTrajOpt#149)",
                ),
            )
        elseif c isa TotalConstraint && _resolves_to_timestep(c.var_name)
            throw(
                ArgumentError(
                    "TotalConstraint on the timestep is obsolete under a time warp: the " *
                    "total duration IS the warp parameter T (never decision data per knot). " *
                    "Pin T via WarpParamBoundsConstraint with lo == hi. (DirectTrajOpt#149)",
                ),
            )
        elseif c isa SymmetryConstraint && c.include_timestep
            throw(
                ArgumentError(
                    "SymmetryConstraint(include_timestep = true) is obsolete under a time " *
                    "warp: the derived timestep is one lattice quantity, not per-knot " *
                    "decision data. (DirectTrajOpt#149)",
                ),
            )
        elseif (c isa BoundsConstraint || c isa EqualityConstraint) &&
               _resolves_to_timestep(c.var_names)
            throw(
                ArgumentError(
                    "bounds/initial/final on the derived timestep component :$dt_name are " *
                    "meaningless under a time warp — Δt is never decision data. The warp's " *
                    "monotonicity enforces Δt > 0; bound the warp parameter(s) instead " *
                    "(WarpParamBoundsConstraint). (DirectTrajOpt#149)",
                ),
            )
        elseif c isa NonlinearKnotPointConstraint && dt_name in c.var_names
            throw(
                ArgumentError(
                    "nonlinear constraint on the derived timestep :$dt_name is meaningless " *
                    "under a time warp — it has no packed decision column. (DirectTrajOpt#149)",
                ),
            )
        end
    end

    # objectives that read the derived timestep as a variable
    sub_objs = obj isa CompositeObjective ? obj.objectives : [obj]
    for o in sub_objs
        if (o isa KnotPointObjective || o isa GlobalKnotPointObjective) &&
           dt_name in o.var_names
            throw(
                ArgumentError(
                    "objective on the derived timestep :$dt_name is meaningless under a " *
                    "time warp — it has no packed decision column. Use " *
                    "MinimumTimeObjective (D·T under a warp). (DirectTrajOpt#149)",
                ),
            )
        end
    end
    return nothing
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrator::AbstractIntegrator;
    kwargs...,
)
    return DirectTrajOptProblem(traj, obj, AbstractIntegrator[integrator]; kwargs...)
end


"""
    get_trajectory_constraints(traj::NamedTrajectory)

Extract and create constraints from a NamedTrajectory's initial, final, and bounds specifications.

# Arguments
- `traj::NamedTrajectory`: Trajectory with specified initial conditions, final conditions, and/or bounds

# Returns
- `Vector{AbstractConstraint}`: Vector of constraints including:
  - Initial value equality constraints (from `traj.initial`)
  - Final value equality constraints (from `traj.final`)
  - Bounds constraints (from `traj.bounds`)

# Details
The function automatically handles time indices based on which constraints are specified:
- If both initial and final constraints exist for a component, bounds apply to interior points (2:N-1)
- If only initial exists, bounds apply from second point onward (2:N)
- If only final exists, bounds apply up to second-to-last point (1:N-1)
- If neither exist, bounds apply to all time points (1:N)
"""
function get_trajectory_constraints(traj::NamedTrajectory)

    cons = AbstractConstraint[]

    # add initial equality constraints
    for (name, val) ∈ pairs(traj.initial)
        con_label = "initial value of $name"
        eq_con = EqualityConstraint(name, [1], val; label = con_label)
        push!(cons, eq_con)
    end

    # add final equality constraints
    for (name, val) ∈ pairs(traj.final)
        label = "final value of $name"
        eq_con = EqualityConstraint(name, [traj.N], val; label = label)
        push!(cons, eq_con)
    end

    # add bounds constraints
    for (name, bound) ∈ pairs(traj.bounds)
        if name ∈ keys(traj.initial) && name ∈ keys(traj.final)
            ts = 2:(traj.N-1)
        elseif name ∈ keys(traj.initial) && !(name ∈ keys(traj.final))
            ts = 2:traj.N
        elseif name ∈ keys(traj.final) && !(name ∈ keys(traj.initial))
            ts = 1:(traj.N-1)
        else
            ts = 1:traj.N
        end
        con_label = "bounds on $name"
        bounds_con = BoundsConstraint(name, ts, bound; label = con_label)
        push!(cons, bounds_con)
    end

    # add time consistency constraint if trajectory has both :t and timestep variable
    timestep_var = traj.timestep
    if timestep_var isa Symbol && :t ∈ traj.names && timestep_var ∈ traj.names
        time_con = TimeConsistencyConstraint(; time_name = :t, timestep_name = timestep_var)
        push!(cons, time_con)

        # add t_1 = 0 constraint if not already specified in initial
        if :t ∉ keys(traj.initial)
            t_init_con = EqualityConstraint(:t, [1], [0.0]; label = "initial time t₁ = 0")
            push!(cons, t_init_con)
        end
    end

    return cons
end

"""
    show_problem_details(io::IO, prob::DirectTrajOptProblem)

Print the trajectory, objective, dynamics, and constraints sections of a problem.

This is used by both `DirectTrajOptProblem` and `QuantumControlProblem` display methods.
"""
function show_problem_details(io::IO, prob::DirectTrajOptProblem)
    traj = prob.trajectory

    # --- Trajectory section ---
    println(io, "  Trajectory")
    println(io, "    Timesteps: ", traj.N)
    println(io, "    Duration:  ", round(get_duration(traj), sigdigits = 6))
    println(io, "    Knot dim:  ", traj.dim)
    vars = join(["$n ($(traj.dims[n]))" for n in traj.names], ", ")
    println(io, "    Variables: ", vars)
    ctrl_str = isempty(traj.control_names) ? "(none)" : join(traj.control_names, ", ")
    println(io, "    Controls:  ", ctrl_str)
    if traj.global_dim > 0
        gvars = join(
            [
                "$n ($(length(traj.global_components[n])))" for
                n in keys(traj.global_components)
            ],
            ", ",
        )
        println(io, "    Globals:   ", gvars)
    end

    # --- Objective section ---
    obj = prob.objective
    if obj isa CompositeObjective
        n = length(obj.objectives)
        println(io, "  Objective ($n terms)")
        for (sub_obj, w) in zip(obj.objectives, obj.weights)
            w_str = string(round(w, sigdigits = 4))
            println(io, "    $(lpad(w_str, 8)) * ", sub_obj)
        end
    elseif obj isa NullObjective
        println(io, "  Objective: NullObjective")
    else
        println(io, "  Objective: ", obj)
    end

    # --- Dynamics section ---
    n_int = length(prob.integrators)
    println(io, "  Dynamics ($n_int integrators)")
    for integ in prob.integrators
        println(io, "    ", integ)
    end

    # --- Constraints section ---
    constraints = prob.constraints
    n_con = length(constraints)
    if n_con > 0
        n_eq = count(c -> c isa EqualityConstraint, constraints)
        n_bnd = count(c -> c isa BoundsConstraint, constraints)
        n_tc = count(c -> c isa TimeConsistencyConstraint, constraints)
        n_other = n_con - n_eq - n_bnd - n_tc

        parts = String[]
        n_eq > 0 && push!(parts, "$n_eq equality")
        n_bnd > 0 && push!(parts, "$n_bnd bounds")
        n_tc > 0 && push!(parts, "$n_tc time consistency")
        n_other > 0 && push!(parts, "$n_other other")

        println(io, "  Constraints ($n_con total: ", join(parts, ", "), ")")
        max_show = 10
        for (i, con) in enumerate(constraints)
            if i <= max_show
                if i < n_con
                    println(io, "    ", con)
                else
                    print(io, "    ", con)
                end
            elseif i == max_show + 1
                print(io, "    ... and $(n_con - max_show) more")
                break
            end
        end
    else
        print(io, "  Constraints: (none)")
    end
end

function Base.show(io::IO, prob::DirectTrajOptProblem)
    println(io, "DirectTrajOptProblem")
    show_problem_details(io, prob)
end

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "DirectTrajOptProblem — integrator convenience forms" setup = [DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    J = QuadraticRegularizer(:u, traj, 1.0)

    # a single AbstractIntegrator is promoted to a vector
    prob_single = DirectTrajOptProblem(traj, J, BilinearIntegrator(G, :x, :u, traj))
    @test prob_single.integrators isa Vector{<:AbstractIntegrator}
    @test length(prob_single.integrators) == 1

    # an empty integrator vector — a dynamics-free constrained problem
    prob_empty = DirectTrajOptProblem(traj, J, AbstractIntegrator[])
    @test isempty(prob_empty.integrators)
    @test !isempty(prob_empty.constraints)  # trajectory constraints still extracted
end

@testitem "DirectTrajOptProblem — default Δt bounds injected when missing" setup =
    [DTOTestHelpers] begin
    N = 5
    comps = (x = randn(4, N), u = randn(2, N), Δt = fill(0.1, 1, N))
    traj = NamedTrajectory(
        comps;
        controls = (:u, :Δt),
        timestep = :Δt,
        initial = (x = zeros(4), u = zeros(2)),
        goal = (x = ones(4),),
    )
    @test !haskey(traj.bounds, :Δt)

    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, AbstractIntegrator[])

    # a zero lower bound was added to prevent negative timesteps
    @test haskey(prob.trajectory.bounds, :Δt)
    @test prob.trajectory.bounds[:Δt][1] == [0.0]
    @test prob.trajectory.bounds[:Δt][2] == [Inf]
end

@testitem "DirectTrajOptProblem — default Δt bounds with global components" setup =
    [DTOTestHelpers] begin
    N = 5
    comps = (x = randn(4, N), u = randn(2, N), Δt = fill(0.1, 1, N))
    gcomps = (g = randn(2),)
    traj = NamedTrajectory(
        comps,
        gcomps;
        controls = (:u, :Δt),
        timestep = :Δt,
        initial = (x = zeros(4), u = zeros(2)),
        goal = (x = ones(4),),
    )
    @test traj.global_dim == 2
    @test !haskey(traj.bounds, :Δt)

    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, AbstractIntegrator[])

    @test haskey(prob.trajectory.bounds, :Δt)
    @test prob.trajectory.bounds[:Δt][1] == [0.0]
    # globals survive the bounds-injection reconstruction
    @test prob.trajectory.global_dim == 2
    @test prob.trajectory.global_data ≈ traj.global_data
end

@testitem "get_trajectory_constraints — final-only bounds span 1:N-1" setup =
    [DTOTestHelpers] begin
    N = 6
    traj = NamedTrajectory(
        (x = randn(4, N), u = randn(2, N), Δt = fill(0.1, 1, N));
        controls = (:u, :Δt),
        timestep = :Δt,
        bounds = (x = (-10.0, 10.0),),
        final = (x = zeros(4),),
    )

    cons = get_trajectory_constraints(traj)
    bnd = only(c for c in cons if c isa BoundsConstraint)
    # x has a final constraint but no initial constraint ⇒ bounds on 1:N-1
    @test bnd.var_names == :x
    @test bnd.times == collect(1:(N-1))

    # and the final equality itself is present
    fin = only(c for c in cons if c isa EqualityConstraint)
    @test fin.var_names == :x
    @test fin.times == [N]
end

@testitem "get_trajectory_constraints — :t consistency constraint and t₁ pinning" setup =
    [DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory(add_time = true)
    @test :t ∈ traj.names
    @test traj.timestep == :Δt

    cons = get_trajectory_constraints(traj)
    @test count(c -> c isa TimeConsistencyConstraint, cons) == 1
    # :t not pinned in initial ⇒ t₁ = 0 equality is added
    @test :t ∉ keys(traj.initial)
    @test length(filter(c -> c isa EqualityConstraint && c.var_names == :t, cons)) == 1

    # with :t pinned in initial, the pin becomes an "initial value of t"
    # EqualityConstraint carrying the pinned value (that IS how pins are
    # enforced) — one :t equality either way, now with the pinned payload.
    N = traj.N
    traj2 = NamedTrajectory(
        (
            x = randn(4, N),
            u = randn(2, N),
            t = collect(0.1:0.1:(0.1N)),
            Δt = fill(0.1, 1, N),
        );
        controls = (:u, :Δt),
        timestep = :Δt,
        initial = (x = zeros(4), u = zeros(2), t = [0.0]),
        bounds = (u = 0.5, Δt = (0.01, 0.5)),
    )
    cons2 = get_trajectory_constraints(traj2)
    @test count(c -> c isa TimeConsistencyConstraint, cons2) == 1
    t_eq = only(filter(c -> c isa EqualityConstraint && c.var_names == :t, cons2))
    @test occursin("initial value of t", t_eq.label)
end

@testitem "DirectTrajOptProblem — user constraints precede trajectory constraints" setup =
    [DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [BilinearIntegrator(G, :x, :u, traj)]
    J = QuadraticRegularizer(:u, traj, 1.0)

    custom = EqualityConstraint(:du, [2], zeros(2); label = "custom eq")
    prob =
        DirectTrajOptProblem(traj, J, integrators; constraints = AbstractConstraint[custom])

    @test prob.constraints[1] === custom
    @test length(prob.constraints) == 1 + length(get_trajectory_constraints(traj))
end

@testitem "show_problem_details — problem display sections" setup = [DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [BilinearIntegrator(G, :x, :u, traj), DerivativeIntegrator(:u, :du, traj)]
    J = TerminalObjective(x -> norm(x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)

    # extra equality constraints push the total past the display truncation limit
    extra = [EqualityConstraint(:du, [k], zeros(2); label = "extra $k") for k = 2:8]
    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = AbstractConstraint[extra...],
    )

    s = sprint(show, prob)
    @test occursin("DirectTrajOptProblem", s)
    @test occursin("Timesteps: $(traj.N)", s)
    @test occursin("Duration:", s)
    @test occursin("Knot dim:", s)
    @test occursin("Globals:", s)
    @test occursin("Objective (3 terms)", s)
    @test occursin("Dynamics (2 integrators)", s)
    # 13 constraints > 10 displayed ⇒ truncation line
    @test occursin("... and ", s)
    @test occursin(" more", s)
    @test occursin("equality", s)
    @test occursin("bounds", s)
end

@testitem "show_problem_details — objective variants and empty sections" setup =
    [DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [BilinearIntegrator(G, :x, :u, traj)]

    # NullObjective section
    prob_null = DirectTrajOptProblem(traj, NullObjective(), integrators)
    @test occursin("Objective: NullObjective", sprint(show, prob_null))

    # single (non-composite) objective section
    prob_single =
        DirectTrajOptProblem(traj, QuadraticRegularizer(:u, traj, 1.0), integrators)
    s_single = sprint(show, prob_single)
    @test occursin("Objective: ", s_single)
    @test !occursin("terms)", s_single)

    # a minimal problem: only the timestep control (NT always promotes the
    # timestep into controls, so "Controls: (none)" is unreachable via the
    # constructor), no constraints, no integrators
    plain = NamedTrajectory(
        (x = randn(4, 5), Δt = fill(0.1, 1, 5));
        controls = :Δt,
        timestep = :Δt,
    )
    prob_plain = DirectTrajOptProblem(plain, NullObjective(), AbstractIntegrator[])
    s_plain = sprint(show, prob_plain)
    @test occursin("Controls:  Δt", s_plain)
    # the default Δt bounds injection guarantees a BoundsConstraint even on a
    # minimal trajectory — "Constraints: (none)" is unreachable via construction
    # (NamedTrajectories types `timestep` as a Symbol, so the injection always
    # applies when the trajectory's own bounds lack the timestep)
    @test occursin("BoundsConstraint: \"bounds on Δt\"", s_plain)
    @test occursin("Dynamics (0 integrators)", s_plain)
end

# ============================================================================ #
# Phase 1b (DTO#149): obsolete time machinery refuses under a warp
# ============================================================================ #

@testitem "obsolete time machinery refuses under a time warp (pointer messages)" begin
    include("../test/test_utils.jl")
    using NamedTrajectories
    using Logging
    using Test

    traj = warped_derivative_trajectory(N = 6, T = 1.0)
    D = DerivativeIntegrator(:u, :du, traj)
    J = QuadraticRegularizer(:u, traj, 1.0)

    # TimeStepsAllEqualConstraint: the uniform mesh is built into the warp lattice
    @test_throws ArgumentError DirectTrajOptProblem(
        traj,
        J,
        [D];
        constraints = AbstractConstraint[TimeStepsAllEqualConstraint()],
    )

    # TimeConsistencyConstraint, user-supplied
    @test_throws ArgumentError DirectTrajOptProblem(
        traj,
        J,
        [D];
        constraints = AbstractConstraint[TimeConsistencyConstraint()],
    )

    # TimeConsistencyConstraint, auto-added because a :t component rides along
    traj_t = NamedTrajectory(
        (
            x = randn(2, 6),
            u = randn(1, 6),
            t = collect(range(0, 1.0, length = 6)),
            Δt = fill(0.2, 1, 6),
        );
        controls = (:u,),
        timestep = :Δt,
        warp = GlobalScale(1.0),
    )
    @test_throws ArgumentError DirectTrajOptProblem(
        traj_t,
        QuadraticRegularizer(:u, traj_t, 1.0),
        AbstractIntegrator[],
    )

    # explicit bounds on the derived timestep: meaningless, refused with a pointer
    traj_b = NamedTrajectory(
        (x = randn(2, 6), u = randn(1, 6), Δt = fill(0.2, 1, 6));
        controls = (:u,),
        timestep = :Δt,
        warp = GlobalScale(1.0),
        bounds = (Δt = (0.01, 0.5),),
    )
    @test_throws ArgumentError DirectTrajOptProblem(
        traj_b,
        QuadraticRegularizer(:u, traj_b, 1.0),
        [D],
    )

    # the auto Δt ≥ 0 injection refuses to run (and must NOT silently drop the
    # warp by rebuilding the trajectory without it) — pointer message, warp kept
    logs = Test.TestLogger()
    prob = with_logger(logs) do
        DirectTrajOptProblem(traj, J, [D])
    end
    @test prob.trajectory.warp === traj.warp
    @test !haskey(prob.trajectory.bounds, :Δt)
    @test any(logs.logs) do rec
        rec.level == Logging.Info && occursin(r"warp|derived"i, rec.message)
    end

    # and the refusal messages carry the pointer to the warp parameterization
    err = try
        DirectTrajOptProblem(
            traj,
            J,
            [D];
            constraints = AbstractConstraint[TimeStepsAllEqualConstraint()],
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(r"warp"i, sprint(showerror, err))
end

@testitem "Bilinear-family integrators refuse warp trajectories (demotion arc)" begin
    include("../test/test_utils.jl")
    using NamedTrajectories
    using Test

    traj = warped_derivative_trajectory(N = 6, T = 1.0)
    G(u) = [0.0 1.0; -0.1 0.0] + u[1] * [0.0 0.0; 0.1 0.0]

    @test_throws ArgumentError BilinearIntegrator(G, :x, :u, traj)
end

end
