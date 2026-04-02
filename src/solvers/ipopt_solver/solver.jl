using DirectTrajOpt
using NamedTrajectories

using MathOptInterface
const MOI = MathOptInterface
using Ipopt
using TestItemRunner
using Libdl  # Added for Pardiso library loading


function DirectTrajOpt._solve(
    prob::DirectTrajOptProblem,
    options::IpoptOptions;
    verbose::Bool = true,
    callback = nothing,
    kwargs...,
)
    # Apply kwargs to matching IpoptOptions fields
    ipopt_fields = fieldnames(IpoptOptions)
    for (k, v) in kwargs
        if k in ipopt_fields
            setfield!(options, k, v)
        else
            @warn "Unknown solver option: $k. Valid options: $(ipopt_fields)"
        end
    end

    # Sync derived fields that depend on other fields.
    # These are computed at IpoptOptions construction time, so kwarg overrides
    # of the source field don't automatically propagate.
    if haskey(kwargs, :eval_hessian) && !haskey(kwargs, :hessian_approximation)
        options.hessian_approximation = options.eval_hessian ? "exact" : "limited-memory"
    end
    if haskey(kwargs, :refine) && !haskey(kwargs, :adaptive_mu_globalization)
        options.adaptive_mu_globalization =
            options.refine ? "obj-constr-filter" : "never-monotone-mode"
    end

    optimizer, variables =
        get_optimizer_and_variables(prob, options, callback, verbose = verbose)

    MOI.optimize!(optimizer)

    update_trajectory!(prob, optimizer, variables)

    return nothing
end


# ----------------------------------------------------------------------------
# Optimizer Initialization/Synchronization
# ----------------------------------------------------------------------------


function get_optimizer_and_variables(
    prob::DirectTrajOptProblem,
    options::IpoptOptions,
    callback::Union{Nothing,Function};
    verbose::Bool = true,
)
    t_init_start = time()
    if verbose
        println("    initializing optimizer...")
    end

    # get evaluator
    t_eval = time()
    evaluator = Solvers.IpoptEvaluator(prob; eval_hessian = options.eval_hessian, verbose = verbose)
    if verbose
        println("    evaluator created ($(round(time() - t_eval, digits=3))s)")
    end

    # get the MOI specific nonlinear constraints
    t_nlcons = time()
    nl_cons = Solvers.get_nonlinear_constraints(prob)
    if verbose
        println(
            "    NL constraint bounds extracted ($(round(time() - t_nlcons, digits=3))s)",
        )
    end

    # build NLP block data
    t_block = time()
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)
    if verbose
        println("    NLP block data built ($(round(time() - t_block, digits=3))s)")
    end

    # initialize optimizer 
    t_opt = time()
    optimizer = Ipopt.Optimizer()

    # set NLP block data
    MOI.set(optimizer, MOI.NLPBlock(), block_data)

    # set objective sense: minimize
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    if verbose
        println("    Ipopt optimizer configured ($(round(time() - t_opt, digits=3))s)")
    end

    # initialize problem variables 
    t_vars = time()
    variables = set_variables!(optimizer, prob.trajectory)
    if verbose
        println("    variables set ($(round(time() - t_vars, digits=3))s)")
    end

    # set callback function
    if !isnothing(callback)
        MOI.set(optimizer, Ipopt.CallbackFunction(), callback(optimizer))
    end

    # add linear constraints
    t_lincons = time()
    linear_constraints = AbstractLinearConstraint[filter(
        c->c isa AbstractLinearConstraint,
        prob.constraints,
    )...]
    Solvers.constrain!(optimizer, variables, linear_constraints, prob.trajectory; verbose = verbose)
    if verbose
        println(
            "    linear constraints added: $(length(linear_constraints)) ($(round(time() - t_lincons, digits=3))s)",
        )
    end

    # set solver options
    set_options!(optimizer, options)

    if verbose
        println(
            "    optimizer initialization complete (total: $(round(time() - t_init_start, digits=3))s)",
        )
    end

    return optimizer, variables
end


function set_variables!(optimizer::Ipopt.Optimizer, traj::NamedTrajectory)
    data_dim = traj.dim * traj.N

    # add variables
    variables = MOI.add_variables(optimizer, data_dim + traj.global_dim)

    # set trajectory data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[1:data_dim],
        collect(traj.datavec),
    )

    # set global data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[data_dim .+ (1:traj.global_dim)],
        collect(traj.global_data),
    )

    return variables
end

function update_trajectory!(
    prob::DirectTrajOptProblem,
    optimizer::Ipopt.Optimizer,
    variables::Vector{MOI.VariableIndex},
)
    update!(
        prob.trajectory,
        MOI.get(optimizer, MOI.VariablePrimal(), variables),
        type = :both,
    )
    return nothing
end


# ----------------------------------------------------------------------------
# Optimizer Configuration/Options
# ----------------------------------------------------------------------------


function DirectTrajOpt.set_options!(optimizer::Ipopt.Optimizer, options::IpoptOptions)
    ignored_options = [:eval_hessian, :refine]

    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name in ignored_options
            continue
        end
        if name == :linear_solver
            if value == "pardiso"
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
            end
        end
        if !isnothing(value)
            optimizer.options[String(name)] = value
        end
    end
    return nothing
end


# ----------------------------------------------------------------------------
# Optimizer Tests
# ----------------------------------------------------------------------------


@testitem "testing Ipopt.jl solver" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(
        u -> [norm(u) - 1.0],
        :u,
        traj;
        times = 2:(traj.N-1),
        equality = false,
    )

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = AbstractConstraint[g_u_norm],
    )

    solve!(prob; max_iter = 100)
end

@testitem "testing Ipopt.jl solver with NonlinearGlobalKnotPointConstraint" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Add global objective - minimize global parameter
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # Knot point constraint with global dependency
    # Couples control magnitude with global parameter
    g_ug = NonlinearGlobalKnotPointConstraint(
        ug -> begin
            u = ug[1:traj.dims[:u]]
            g = ug[(traj.dims[:u]+1):end]
            return [norm(u) * (1.0 + norm(g)) - 2.0]
        end,
        [:u],
        [:g],
        traj;
        times = 2:(traj.N-1),
        equality = false,
    )

    prob =
        DirectTrajOptProblem(traj, J, integrators; constraints = AbstractConstraint[g_ug])

    solve!(prob; max_iter = 100)

    # Verify constraint is satisfied at each timestep
    for k = 2:(traj.N-1)
        u = traj[k][:u]
        g = traj.global_data[traj.global_components[:g]]
        @test norm(u) * (1.0 + norm(g)) <= 2.0 + 1e-6
    end
end

@testitem "testing Ipopt.jl solver with NonlinearGlobalConstraint" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Add global objective - minimize global parameter
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 10.0)

    # Pure global constraint - bounds the global parameter
    g_global = NonlinearGlobalConstraint(g -> [norm(g) - 0.5], :g, traj; equality = false)

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = AbstractConstraint[g_global],
    )

    solve!(prob; max_iter = 100)

    # Verify global variable is within constraint
    @test norm(traj.global_data[traj.global_components[:g]]) <= 0.5 + 1e-6
end

# ============================================================================= #
# Tests for solve! kwargs interface
# ============================================================================= #

@testitem "solve! kwargs applied to IpoptOptions" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators)

    # Multiple kwargs: max_iter and tol
    solve!(prob; max_iter = 10, tol = 1e-4)
end

@testitem "solve! kwargs override options struct" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators)

    # options struct says max_iter=10, kwarg overrides to 50
    solve!(prob; options = IpoptOptions(max_iter = 10), max_iter = 50)
end

@testitem "solve! warns on unknown kwargs" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators)

    # Unknown kwarg should produce a warning
    @test_logs (:warn, r"Unknown solver option: bad_opt") solve!(
        prob;
        max_iter = 10,
        bad_opt = 42,
    )
end

@testitem "solve! eval_hessian as kwarg" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators)

    # eval_hessian=false triggers L-BFGS approximation
    solve!(prob; max_iter = 10, eval_hessian = false)
end

@testitem "solve! with previously buried Ipopt option" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators)

    # mu_strategy was previously only accessible via IpoptOptions struct
    solve!(prob; mu_strategy = "monotone", max_iter = 10)
end

@testitem "solve! syncs derived options from eval_hessian and refine kwargs" begin
    include("../../../test/test_utils.jl")

    # eval_hessian → hessian_approximation
    opts = IpoptOptions()
    @test opts.eval_hessian == true
    @test opts.hessian_approximation == "exact"

    # setfield! alone does NOT update derived field (struct-level limitation)
    opts2 = IpoptOptions()
    setfield!(opts2, :eval_hessian, false)
    @test opts2.hessian_approximation == "exact"

    # IpoptOptions constructor correctly computes derived field
    opts3 = IpoptOptions(eval_hessian = false)
    @test opts3.hessian_approximation == "limited-memory"

    # refine → adaptive_mu_globalization
    opts4 = IpoptOptions(refine = false)
    @test opts4.adaptive_mu_globalization == "never-monotone-mode"

    # Explicit override takes precedence over sync
    opts5 = IpoptOptions(eval_hessian = false, hessian_approximation = "exact")
    @test opts5.hessian_approximation == "exact"

    # End-to-end: solve! with eval_hessian=false kwarg should use L-BFGS
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    prob = DirectTrajOptProblem(traj, J, integrators)
    solve!(prob; max_iter = 5, eval_hessian = false, print_level = 0)
end
