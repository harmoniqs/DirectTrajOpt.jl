import Ipopt, MadNLP
using DirectTrajOpt

include("DirectTrajOpt.jl/test/solver_test_utils.jl")

prob = get_seeded_prob(42)
optim, vars = MadNLPSolverExt.get_optimizer_and_variables(prob, MadNLPSolverExt.MadNLPOptions(; max_iter=100), nothing)

# MadNLP.optimize!(optim)

const MadNLPMOI = [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPMOI][1]



@assert optim.solver isa Nothing
begin
    # # Custom features
    # @assert !isempty(optim.variables)
    # @assert optim.nlp_model isa Nothing # assumes the NLP is specified exclusively via `nlp_data`
    # @assert isempty(optim.vector_nonlinear_oracle_constraints)
    # @assert !in(_kFunctionTypeScalarQuadratic, optim.qp_data.function_type)

    # has_nlp_constraints = !isempty(optim.nlp_data.constraint_bounds) # default: true
    # has_nlp_objective = optim.nlp_data.has_objective # default: true
    # has_hessian = in(:Hess, MOI.features_available(optim.nlp_data.evaluator)) # default: true
    # has_jacobian_operator = in(:JacVec, MOI.features_available(optim.nlp_data.evaluator)) # default: false
    # has_hessian_operator = in(:HessVec, MOI.features_available(optim.nlp_data.evaluator)) # default: false

    # optim.has_only_linear_constraints = !has_nlp_constraints # default: false
    # optim.islp = optim.has_only_linear_constraints && !has_nlp_objective # default: false
    # optim.jprod_available = has_jacobian_operator # default: false
    # optim.hprod_available = has_hessian_operator # default: false
    # optim.hess_available = has_hessian # default: true

    # Default features
    @assert !isempty(optim.variables)
    @assert optim.nlp_model isa Nothing # assumes the NLP is specified exclusively via `nlp_data`
    @assert isempty(optim.vector_nonlinear_oracle_constraints)
    @assert !in(_kFunctionTypeScalarQuadratic, optim.qp_data.function_type)

    @assert !isempty(optim.nlp_data.constraint_bounds)
    @assert optim.nlp_data.has_objective
    @assert in(:Grad, MOI.features_available(optim.nlp_data.evaluator))
    @assert in(:Jac, MOI.features_available(optim.nlp_data.evaluator))
    @assert in(:Hess, MOI.features_available(optim.nlp_data.evaluator))
    @assert !in(:JacVec, MOI.features_available(optim.nlp_data_evaluator))
    @assert !in(:HessVec, MOI.features_available(optim.nlp_data.evaluator))

    optim.has_only_linear_constraints = false
    optim.islp = false
    optim.jprod_available = false
    optim.hprod_available = false
    optim.hess_available = true
    MOI.initialize(optim.nlp_data.evaluator, [:Grad, :Jac, :Hess])
    
    # Sparsity
    jacobian_sparsity = MOI.jacobian_structure(optim)
    nnzj = length(jacobian_sparsity)
    jrows = Vector{Int}(undef, nnzj)
    jcols = Vector{Int}(undef, nnzj)
    for k in 1:nnzj
        jrows[k], jcols[k] = jacobian_sparsity[k]
    end
    optim.jrows = jrows
    ooptim.jcols = jcols

    hessian_sparsity = MOI.hessian_lagrangian_structure(optim)
    nnzh = length(hessian_sparsity)
    hrows = Vector{Int}(undef, nnzh)
    hcols = Vector{Int}(undef, nnzh)
    for k in 1:nnzj
        hrows[k], hcols[k] = hessian_sparsity[k]
    end
    optim.hrows = hrows
    optim.hcols = hcols

    optim.needs_new_nlp = true

    #

    @assert !optim.invalid_model
    array_type = pop!(optim.options, :array_type, nothing)

    #

    nnzj = length(optim.jrows)
    nnzh = length(optim.hrows)

    @assert !in(nothing, optim.variable_primal_start)
    nvar = length(optim.variables.lower)
    x0 = zeros(Float64, nvar)
    x0 .= optim.variable_primal_start

    g_L, g_U = copy(optim.qp_data.g_L), copy(optim.qp_data.g_U)
    for bound = optim.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    ncon = length(g_L)

    y0 = zeros(Float64, ncon)
    for (i, start) = enumerate(optim.qp_data.mult_g)
        y0[i] = _dual_start(optim, start, -1) # *
    end
    offset = length(optim.qp_data.mult_g)
    if optim.nlp_dual_start isa Nothing
        for (k, v) = optim.mult_g_nlp
            # y0[offset + k.value] = _dual_start(optim, v, -1)
            y0[offset + k.value] = ((v isa Nothing) ? 0. : (((optim.sense == MOI.MIN_SENSE) ? -1.0 : 1.0) * v))
        end
    else
        for (i, start) = enumerate(optim.nlp_dual_start::Vector{Float64})
            y0[offset + i] = ((optim.sense == MOI.MIN_SENSE) ? -1.0 : 1.0) * start
        end
    end

    nlp = MadNLPMOI.MOIModel(
        MadNLP.NLPModels.NLPModelMeta(
            nvar;
            x0=x0, lvar=optim.variables.lower, uvar=optim.variables.upper,
            ncon=ncon, y0=y0, lcon=g_L, ucon=g_U,
            nnzj=nnzj, nnzh=nnzh, minimize = (optim.sense == MOI.MIN_SENSE), islp=optim.islp,
            name="MOIModel", variable_bounds_analysis=false, constraint_bounds_analysis=false,
            jprod_available=optim.jprod_available, hprod_available=optim.hprod_available, hess_available=optim.hess_available,
        ),
        optim,
        NLPModels.Counters(),
    )

end
@assert optim.nlp_model isa Nothing
@assert optim.silent || !optim.options[:print_level] == MadNLP.ERROR
@assert optim.hess_available || optim.options[:hessian_approximation] == MadNLP.CompactLBFGS
@assert !optim.has_only_linear_constraints || optim.options[:jacobian_constant] == true
@assert length(optim.vector_nonlinear_oracle_constraints) == 0 # otherwise must reset timers for each constraint
optim.solver = MadNLP.MadNLPSolver(optim.nlp; optim.options...)
result = MadNLP.solve!(optim.solver)
optim.result = ((array_type isa Nothing) ? result : copy_result_to_cpu(result))
optim.solve_time = optim.solver.cnt.total_time
optim.solve_iterations = optim.solver.cnt.k


using DirectTrajOpt
using Piccolo
using Piccolissimo
using LinearAlgebra

function ket_problem()
    T = 10.0
    N = 20
    sys = QuantumSystem(GATES.Z, [GATES.X, GATES.Y], [1.0, 1.0])
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]

    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, T)
    integrator = HermitianExponentialIntegrator(qtraj, N)

    @assert integrator isa HermitianExponentialIntegrator{KetTrajectory}

    qcp = SmoothPulseProblem(qtraj, N; Q = 50.0, R = 1e-3, integrator = [integrator])

    @assert qcp isa QuantumControlProblem
    @assert length(qcp.prob.integrators) == 3  # dynamics + du + ddu
    @assert qcp.prob.integrators[1] isa HermitianExponentialIntegrator{KetTrajectory}

    # solve!(qcp; max_iter = 220)
    return qcp

end

