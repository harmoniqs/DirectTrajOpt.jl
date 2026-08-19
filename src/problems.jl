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
    if timestep_var isa Symbol && !haskey(traj.bounds, timestep_var)
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
    return DirectTrajOptProblem(traj, obj, integrators, all_constraints)
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
    comps = (x = randn(4, N), u = randn(2, N), Δt = fill(0.1, N))
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
    comps = (x = randn(4, N), u = randn(2, N), Δt = fill(0.1, N))
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
        (x = randn(4, N), u = randn(2, N));
        controls = :u,
        timestep = 0.1,
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
        (x = randn(4, N), u = randn(2, N), t = collect(0.1:0.1:(0.1N)), Δt = fill(0.1, N));
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
    @test occursin("Constraints: (none)", s_plain)
    @test occursin("Dynamics (0 integrators)", s_plain)
end

end
