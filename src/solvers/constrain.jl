using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils
using SparseArrays

using DirectTrajOpt.Constraints


function constrain!(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{<:AbstractLinearConstraint},
    traj::NamedTrajectory;
    verbose = false,
)
    for con! ∈ cons
        if verbose
            println("        applying constraint: ", con!.label)
        end
        # Apply constraint to optimizer (computes indices internally)
        con!(opt, vars, traj)
    end
    return nothing
end

function (con::EqualityConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    name = con.var_names

    if con.is_global
        # Global variable constraint
        @assert name ∈ traj.global_names "Global variable $name not found in trajectory"
        @assert length(con.values) == traj.global_dims[name] "Value dimension mismatch for global variable $name"

        indices = WarpPlumbing.packed_globals_base(traj) .+ traj.global_components[name]

        for (i, val) ∈ zip(indices, con.values)
            MOI.add_constraints(opt, vars[i], MOI.EqualTo(val))
        end
    else
        # Trajectory variable constraint
        @assert name ∈ traj.names "Variable $name not found in trajectory"
        # Narrow `ts` from Union{Nothing, Vector{Int}}. In the not-global branch
        # `times` is always a Vector{Int} (set by the trajectory-variable
        # constructors); the explicit `=== nothing` branch lets JET prove that
        # the surviving binding is `Vector{Int}` without flagging the impossible
        # `convert(Vector{Int}, Nothing)` path that a bare type assert produces.
        ts = con.times
        ts === nothing &&
            error("EqualityConstraint: times must be set for a non-global variable")

        if con.values isa Matrix{Float64}
            # Per-knot values: column k → knot ts[k]
            @assert size(con.values, 1) == traj.dims[name] (
                "Matrix row dimension ($(size(con.values, 1))) must match variable dimension ($(traj.dims[name])) for $name"
            )
            for (k, t) ∈ enumerate(ts)
                indices = WarpPlumbing.packed_slice(traj, t, traj.components[name])
                for (i, val) ∈ zip(indices, @view con.values[:, k])
                    MOI.add_constraints(opt, vars[i], MOI.EqualTo(val))
                end
            end
        else
            # Uniform values (existing behavior)
            if length(con.values) == 1
                val_per_time = fill(con.values[1], traj.dims[name])
            else
                @assert length(con.values) == traj.dims[name] "Value dimension mismatch for variable $name"
                val_per_time = con.values
            end

            for t ∈ ts
                indices = WarpPlumbing.packed_slice(traj, t, traj.components[name])
                for (i, val) ∈ zip(indices, val_per_time)
                    MOI.add_constraints(opt, vars[i], MOI.EqualTo(val))
                end
            end
        end
    end
end

function (con::BoundsConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    name = con.var_names
    bounds_val = con.bounds_values

    if con.is_global
        # Global variable constraint
        @assert name ∈ traj.global_names "Global variable $name not found in trajectory"
        var_dim = traj.global_dims[name]

        # Convert bounds to (lb, ub) tuple vectors
        if bounds_val isa Float64
            @assert bounds_val >= 0 "Scalar bound must be non-negative"
            lb = fill(-bounds_val, var_dim)
            ub = fill(bounds_val, var_dim)
        elseif bounds_val isa Vector{Float64}
            @assert length(bounds_val) == var_dim "Bound dimension mismatch"
            @assert all(bounds_val .>= 0) "Vector bound must be non-negative"
            lb = -bounds_val
            ub = bounds_val
        else  # Tuple
            lb, ub = bounds_val
            @assert length(lb) == length(ub) == var_dim "Bound dimension mismatch"
            @assert all(lb .<= ub) "Lower bounds must be <= upper bounds"
        end

        indices = WarpPlumbing.packed_globals_base(traj) .+ traj.global_components[name]

        for (i, (lb_i, ub_i)) ∈ zip(indices, zip(lb, ub))
            MOI.add_constraints(opt, vars[i], MOI.GreaterThan(lb_i))
            MOI.add_constraints(opt, vars[i], MOI.LessThan(ub_i))
        end
    else
        # Trajectory variable constraint
        @assert name ∈ traj.names "Variable $name not found in trajectory"
        ts = con.times
        ts === nothing &&
            error("BoundsConstraint: times must be set for a non-global variable")
        var_dim = traj.dims[name]

        # Determine subcomponents to constrain
        subcomps = isnothing(con.subcomponents) ? (1:var_dim) : con.subcomponents

        # Convert bounds to (lb, ub) tuple vectors
        if bounds_val isa Float64
            @assert bounds_val >= 0 "Scalar bound must be non-negative"
            lb = fill(-bounds_val, var_dim)
            ub = fill(bounds_val, var_dim)
        elseif bounds_val isa Vector{Float64}
            @assert length(bounds_val) == var_dim "Bound dimension mismatch"
            @assert all(bounds_val .>= 0) "Vector bound must be non-negative"
            lb = -bounds_val
            ub = bounds_val
        else  # Tuple
            lb, ub = bounds_val
            @assert length(lb) == length(ub) == var_dim "Bound dimension mismatch"
            @assert all(lb .<= ub) "Lower bounds must be <= upper bounds"
        end

        # Apply bounds at each knot (only for selected subcomponents)
        for t ∈ ts
            indices = WarpPlumbing.packed_slice(traj, t, traj.components[name][subcomps])
            for (i, (lb_i, ub_i)) ∈ zip(indices, zip(lb[subcomps], ub[subcomps]))
                MOI.add_constraints(opt, vars[i], MOI.GreaterThan(lb_i))
                MOI.add_constraints(opt, vars[i], MOI.LessThan(ub_i))
            end
        end
    end
end

function (con::AllEqualConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    # Phase 1b (DTO#149): obsolete under a warp — refused at problem construction
    # already; this is defense in depth for hand-assembled problems.
    traj.warp !== nothing &&
        (con.var_name == :Δt || con.var_name == traj.timestep) &&
        throw(
            ArgumentError(
                "AllEqualConstraint on the timestep is obsolete under a time warp — the " *
                "uniform mesh is built into the warp lattice and the duration is the warp " *
                "parameter T (bound it via WarpParamBoundsConstraint). (DirectTrajOpt#149)",
            ),
        )
    # Determine which variable to constrain (use trajectory's timestep if :Δt)
    var_name = con.var_name == :Δt ? traj.timestep : con.var_name
    @assert var_name isa Symbol "Trajectory must have a timestep variable for AllEqualConstraint"
    @assert var_name ∈ traj.names "Variable $var_name not found in trajectory"

    comp_idx = con.component_index
    @assert comp_idx <= traj.dims[var_name] "Component index $comp_idx exceeds variable dimension"

    # All timesteps 1:N-1 must equal timestep N
    comp = traj.components[var_name][comp_idx]
    indices = [WarpPlumbing.packed_row_index(traj, k, comp) for k ∈ 1:(traj.N-1)]
    bar_index = WarpPlumbing.packed_row_index(traj, traj.N, comp)

    x_minus_val = MOI.ScalarAffineTerm(-1.0, vars[bar_index])
    for i ∈ indices
        xᵢ = MOI.ScalarAffineTerm(1.0, vars[i])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([xᵢ, x_minus_val], 0.0),
            MOI.EqualTo(0.0),
        )
    end
end

function (con::L1SlackConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    v_comps = traj.components[con.var_name]
    s_comps = traj.components[con.slack_name]

    for t ∈ con.times
        v_indices = WarpPlumbing.packed_slice(traj, t, v_comps)
        s_indices = WarpPlumbing.packed_slice(traj, t, s_comps)

        for (vi, si) ∈ zip(v_indices, s_indices)
            # v_{k,i} - s_{k,i} ≤ 0
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction(
                    [
                        MOI.ScalarAffineTerm(1.0, vars[vi]),
                        MOI.ScalarAffineTerm(-1.0, vars[si]),
                    ],
                    0.0,
                ),
                MOI.LessThan(0.0),
            )
            # -v_{k,i} - s_{k,i} ≤ 0
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction(
                    [
                        MOI.ScalarAffineTerm(-1.0, vars[vi]),
                        MOI.ScalarAffineTerm(-1.0, vars[si]),
                    ],
                    0.0,
                ),
                MOI.LessThan(0.0),
            )
        end
    end
end

function (con::TotalConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    # Determine which variable to sum (use trajectory's timestep if :Δt)
    var_name = con.var_name == :Δt ? traj.timestep : con.var_name
    @assert var_name isa Symbol "Trajectory must have a timestep variable for TotalConstraint"
    @assert var_name ∈ traj.names "Variable $var_name not found in trajectory"

    # Phase 1b (DTO#149): obsolete on the derived timestep under a warp.
    traj.warp !== nothing &&
        var_name == traj.timestep &&
        throw(
            ArgumentError(
                "TotalConstraint on the timestep is obsolete under a time warp — the total " *
                "duration IS the warp parameter T; pin it via WarpParamBoundsConstraint with " *
                "lo == hi. (DirectTrajOpt#149)",
            ),
        )
    comp_idx = con.component_index
    @assert comp_idx <= traj.dims[var_name] "Component index $comp_idx exceeds variable dimension"

    # For timestep variables, sum only first N-1 (last knot point has no duration after it)
    # For other variables, sum all N values
    time_indices = (var_name == traj.timestep) ? (1:(traj.N-1)) : (1:traj.N)
    comp = traj.components[var_name][comp_idx]
    indices = [WarpPlumbing.packed_row_index(traj, k, comp) for k ∈ time_indices]

    MOI.add_constraints(
        opt,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, vars[idx]) for idx in indices],
            0.0,
        ),
        MOI.EqualTo(con.value),
    )
end

function (con::SymmetryConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    @assert con.var_name ∈ traj.names "Variable $(con.var_name) not found in trajectory"

    even_pairs = Vector{Tuple{Int,Int}}()
    odd_pairs = Vector{Tuple{Int,Int}}()

    # Get component indices for the variable
    component_indices = [
        WarpPlumbing.packed_slice(traj, t, traj.components[con.var_name])[con.component_indices]
        for t ∈ 1:traj.N
    ]

    if con.even
        even_pairs = vcat(
            even_pairs,
            reduce(
                vcat,
                [
                    collect(zip(component_indices[[idx, traj.N - idx + 1]]...)) for
                    idx = 1:(traj.N÷2)
                ],
            ),
        )
    else
        odd_pairs = vcat(
            odd_pairs,
            reduce(
                vcat,
                [
                    collect(zip(component_indices[[idx, traj.N - idx + 1]]...)) for
                    idx = 1:(traj.N÷2)
                ],
            ),
        )
    end

    # Add timestep symmetry if requested and timestep exists
    if con.include_timestep && traj.timestep isa Symbol
        traj.warp !== nothing && throw(
            ArgumentError(
                "SymmetryConstraint(include_timestep = true) is obsolete under a time warp " *
                "— the derived timestep is one lattice quantity, not per-knot decision " *
                "data. (DirectTrajOpt#149)",
            ),
        )
        time_indices = [
            WarpPlumbing.packed_row_index(traj, k, traj.components[traj.timestep][1])
            for k ∈ 1:traj.N
        ]
        even_pairs = vcat(
            even_pairs,
            [(time_indices[idx], time_indices[traj.N+1-idx]) for idx ∈ 1:(traj.N÷2)],
        )
    end

    # Add even symmetry constraints: x[t] = x[N-t+1]
    for (i1, i2) in even_pairs
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, vars[i1]), MOI.ScalarAffineTerm(-1.0, vars[i2])],
                0.0,
            ),
            MOI.EqualTo(0.0),
        )
    end

    # Add odd symmetry constraints: x[t] = -x[N-t+1]
    for (i1, i2) in odd_pairs
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, vars[i1]), MOI.ScalarAffineTerm(1.0, vars[i2])],
                0.0,
            ),
            MOI.EqualTo(0.0),
        )
    end
end

function (con::GlobalLinearConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    haskey(traj.global_components, con.name) ||
        error("GlobalLinearConstraint: global :$(con.name) not found in trajectory")
    g = traj.global_components[con.name]          # indices within global_data
    size(con.A, 2) == length(g) || error(
        "GlobalLinearConstraint: A has $(size(con.A, 2)) columns but global " *
        ":$(con.name) has dim $(length(g))",
    )

    base = WarpPlumbing.packed_globals_base(traj)  # global vars follow the knot vars
    nrows = size(con.A, 1)

    # Bucket the sparse entries by row (CSC is column-major, so collect once).
    row_terms = [MOI.ScalarAffineTerm{Float64}[] for _ = 1:nrows]
    Is, Js, Vs = findnz(con.A)
    for (i, j, v) in zip(Is, Js, Vs)
        push!(row_terms[i], MOI.ScalarAffineTerm(v, vars[base+g[j]]))
    end

    for r = 1:nrows
        lo, hi = con.lb[r], con.ub[r]
        if isempty(row_terms[r])
            # All-zero row: A[r,:]·g ≡ 0, so the row reduces to lo ≤ 0 ≤ hi.
            # Feasible iff 0 ∈ [lo, hi] — then there is nothing to add. Otherwise
            # the user specified a structurally infeasible row; surface it rather
            # than silently dropping it.
            lo <= 0 <= hi || error(
                "GlobalLinearConstraint: row $r of A is all zeros, reducing to " *
                "$lo ≤ 0 ≤ $hi, which is infeasible",
            )
            continue
        end
        f = MOI.ScalarAffineFunction(row_terms[r], 0.0)
        if lo == hi
            MOI.add_constraints(opt, f, MOI.EqualTo(lo))
        else
            isfinite(lo) && MOI.add_constraints(opt, f, MOI.GreaterThan(lo))
            isfinite(hi) && MOI.add_constraints(opt, f, MOI.LessThan(hi))
        end
    end
    return nothing
end

function (con::TimeConsistencyConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    # Get variable names, using trajectory's timestep if :Δt is specified
    time_name = con.time_name
    timestep_name = con.timestep_name == :Δt ? traj.timestep : con.timestep_name

    @assert time_name ∈ traj.names "Time variable $time_name not found in trajectory"
    @assert timestep_name isa Symbol "Trajectory must have a timestep variable"
    @assert timestep_name ∈ traj.names "Timestep variable $timestep_name not found in trajectory"

    # Phase 1b (DTO#149): obsolete under a warp — the Δt rows are derived, so the
    # consistency holds by construction. Defense in depth behind the problem-level
    # refusal.
    traj.warp !== nothing &&
        timestep_name == traj.timestep &&
        throw(
            ArgumentError(
                "TimeConsistencyConstraint is obsolete under a time warp — the timestep rows " *
                "are derived from the warp, so tₖ₊₁ = tₖ + Δtₖ holds by construction. Physical " *
                "knot times are knot_times(traj.warp, traj.N). (DirectTrajOpt#149)",
            ),
        )
    # For each k = 1:N-1, add constraint: t_{k+1} - t_k - Δt_k = 0
    for k = 1:(traj.N-1)
        t_k = WarpPlumbing.packed_row_index(traj, k, traj.components[time_name][1])
        t_k1 = WarpPlumbing.packed_row_index(traj, k+1, traj.components[time_name][1])
        Δt_k = WarpPlumbing.packed_row_index(traj, k, traj.components[timestep_name][1])

        # t_{k+1} - t_k - Δt_k = 0
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, vars[t_k1]),   # + t_{k+1}
                    MOI.ScalarAffineTerm(-1.0, vars[t_k]),   # - t_k
                    MOI.ScalarAffineTerm(-1.0, vars[Δt_k]),   # - Δt_k
                ],
                0.0,
            ),
            MOI.EqualTo(0.0),
        )
    end
end

function (con::WarpParamBoundsConstraint)(
    opt::AbstractOptimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory,
)
    warp = traj.warp
    warp === nothing && throw(
        ArgumentError(
            "WarpParamBoundsConstraint requires a trajectory with a time warp — there are " *
            "no warp parameters to bound (the per-knot timestep is an ordinary variable " *
            "there; bound it with BoundsConstraint). (DirectTrajOpt#149)",
        ),
    )
    length(con.lo) == n_params(warp) || throw(
        ArgumentError(
            "WarpParamBoundsConstraint bounds length $(length(con.lo)) does not match the " *
            "warp parameter count $(n_params(warp))",
        ),
    )
    for (j, pos) ∈ enumerate(WarpPlumbing.warp_param_indices(traj))
        MOI.add_constraints(opt, vars[pos], MOI.GreaterThan(con.lo[j]))
        MOI.add_constraints(opt, vars[pos], MOI.LessThan(con.hi[j]))
    end
    return nothing
end


# Coverage targets: src/solvers/constrain.jl (92% → ~95%) + cross-solver agreement

@testitem "constrain! verbose output" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, integrators)

    evaluator = Solvers.Evaluator(prob; eval_hessian = true, verbose = false)
    nl_cons = Solvers.get_nonlinear_constraints(prob)
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)

    optimizer = Ipopt.Optimizer()
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    data_dim = traj.dim * traj.N
    variables = MOI.add_variables(optimizer, data_dim + traj.global_dim)
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[1:data_dim],
        collect(traj.datavec),
    )

    linear_constraints = AbstractLinearConstraint[filter(
        c->c isa AbstractLinearConstraint,
        prob.constraints,
    )...]

    output = capture_stdout() do
        Solvers.constrain!(
            optimizer,
            variables,
            linear_constraints,
            prob.trajectory;
            verbose = true,
        )
    end
    @test contains(output, "applying constraint")
end

@testitem "Cross-solver agreement (Ipopt vs MadNLP)" setup=[DTOTestHelpers] begin
    using Random

    include(
        joinpath(dirname(dirname(pathof(DirectTrajOpt))), "test", "solver_test_utils.jl"),
    )

    seed = UInt64(42)

    prob_ipopt = get_seeded_prob_solved(
        seed,
        IpoptSolverExt.IpoptOptions(; max_iter = 100, print_level = 0),
    )
    prob_madnlp =
        get_seeded_prob_solved(seed, DirectTrajOpt.MadNLPOptions(; max_iter = 100))

    traj_ipopt = prob_ipopt.trajectory
    traj_madnlp = prob_madnlp.trajectory

    traj_dist = (traj_madnlp.data[:, :] .- traj_ipopt.data[:, :]) .^ 2
    traj_dist = sqrt(sum(traj_dist)) / length(traj_dist)

    @test traj_dist < 1e-4
end

@testitem "coverage: BoundsConstraint vector and tuple bound application" setup =
    [DTOTestHelpers] begin
    _, traj = bilinear_dynamics_and_trajectory(add_global = true)

    n_vars = traj.dim * traj.N + traj.global_dim
    g_dim = length(traj.global_components[:g])

    function apply_to_fresh_optimizer(con)
        opt = Ipopt.Optimizer()
        vars = MOI.add_variables(opt, n_vars)
        con(opt, vars, traj)
        return opt
    end

    n_greater_than(opt) =
        MOI.get(opt, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.GreaterThan{Float64}}())
    n_less_than(opt) =
        MOI.get(opt, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.LessThan{Float64}}())

    # Vector{Float64} bounds on a global variable: symmetric [-b, b] per component
    b = 0.1 .+ 0.01 .* collect(1:g_dim)
    opt = apply_to_fresh_optimizer(GlobalBoundsConstraint(:g, b))
    @test n_greater_than(opt) == g_dim
    @test n_less_than(opt) == g_dim

    # (lb, ub) tuple bounds on a global variable
    lb = fill(-0.2, g_dim)
    ub = fill(0.3, g_dim)
    opt = apply_to_fresh_optimizer(GlobalBoundsConstraint(:g, (lb, ub)))
    @test n_greater_than(opt) == g_dim
    @test n_less_than(opt) == g_dim

    # Vector{Float64} bounds on a trajectory variable (du has dim 2)
    du_dim = traj.dims[:du]
    opt = apply_to_fresh_optimizer(BoundsConstraint(:du, 1:traj.N, fill(0.4, du_dim)))
    @test n_greater_than(opt) == du_dim * traj.N
    @test n_less_than(opt) == du_dim * traj.N
end

@testitem "coverage: GlobalLinearConstraint skips feasible all-zero rows" setup =
    [DTOTestHelpers] begin
    using SparseArrays

    _, traj = bilinear_dynamics_and_trajectory(add_global = true)

    g_dim = length(traj.global_components[:g])
    # Row 1 pins g[1] - g[2] = 0; row 2 is all zeros with 0 ∈ [lo, hi] —
    # structurally feasible, so it is skipped (continue) rather than an error.
    A = spzeros(2, g_dim)
    A[1, 1] = 1.0
    A[1, 2] = -1.0
    con = GlobalLinearConstraint(:g, A, [0.0, -1.0], [0.0, 1.0])

    # A mock optimizer: Ipopt's MOI wrapper does not implement
    # NumberOfConstraints for affine-in-set constraints, and the functor only
    # needs add_constraints.
    opt = MOI.Utilities.MockOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    )
    vars = MOI.add_variables(opt, traj.dim * traj.N + traj.global_dim)
    con(opt, vars, traj)

    # Only the equality row was materialized (one affine-in-EqualTo constraint).
    @test MOI.get(
        opt,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}(),
    ) == 1
end
