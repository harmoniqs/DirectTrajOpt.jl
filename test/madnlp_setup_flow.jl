import Ipopt, MadNLP
using DirectTrajOpt

include("DirectTrajOpt.jl/test/solver_test_utils.jl")

prob = get_seeded_prob(42)
optim, vars = MadNLPSolverExt.get_optimizer_and_variables(prob, MadNLPSolverExt.MadNLPOptions(; max_iter=100), nothing)

# MadNLP.optimize!(optim)

const MadNLPMOI = [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPMOI][1]


@assert optim.solver isa Nothing
MadNLPMOI._setup_model(optim)
@assert !optim.invalid_model
array_type = pop!(optim.options, :array_type, nothing)
MadNLPMOI._setup_nlp(optim; array_type=nothing) # array_type comes from pop!(optim, :array_type, nothing)
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

###

# Helper function for dumping/inspecting a Julia value

function dump_obj(obj; k_colsz=nothing, v_colsz=nothing, repr_fn=nothing)
    # show_pair(k, v, m) = println("$(k):$(join(fill(" ", m - length(k))))$(v)")
    show_pair(k, v, m; l=nothing) = println("$(k):$(join(fill(" ", m - length(k))))$((l isa Nothing) ? v : (length(v) <= l ? v : join([v[1:min(length(v), l)], "..."])))")

    # Given `obj::Any`:
    typ = typeof(obj)
    ks = fieldnames(typ)
    vs = ((repr_fn isa Nothing) ? repr : repr_fn).((k -> getfield(obj, k)).(ks))

    m = ((k_colsz isa Nothing) ? 3 : k_colsz) + maximum(length.(String.(ks)))
    l = (v_colsz isa Nothing) ? nothing : v_colsz
    
    for (k, v) = zip(ks, vs)
        show_pair(String(k), v, m; l=l)
    end

    return nothing
end

###

# MOI_wrapper.jl line 1303

function _setup_model(model::Optimizer)
    vars = MOI.get(model.variables, MOI.ListOfVariableIndices())
    if isempty(vars)
        model.invalid_model = true
        return
    end
    # Create NLP backend.
    if model.nlp_model !== nothing
        evaluator = MOI.Nonlinear.Evaluator(model.nlp_model, model.ad_backend, vars)
        model.nlp_data = MOI.NLPBlockData(evaluator)
    end
    # Check model's structure.
    has_oracle = !isempty(model.vector_nonlinear_oracle_constraints)
    has_quadratic_constraints =
        any(isequal(_kFunctionTypeScalarQuadratic), model.qp_data.function_type)
    has_nlp_constraints = !isempty(model.nlp_data.constraint_bounds) || has_oracle
    has_nlp_objective = model.nlp_data.has_objective
    has_hessian = :Hess in MOI.features_available(model.nlp_data.evaluator)
    has_jacobian_operator = :JacVec in MOI.features_available(model.nlp_data.evaluator)
    has_hessian_operator = :HessVec in MOI.features_available(model.nlp_data.evaluator)
    for (_, s) in model.vector_nonlinear_oracle_constraints
        if s.set.eval_hessian_lagrangian === nothing
            has_hessian = false
            break
        end
    end

    model.has_only_linear_constraints = !has_quadratic_constraints && !has_nlp_constraints
    model.islp = model.has_only_linear_constraints && !has_nlp_objective
    model.jprod_available = has_jacobian_operator && !has_oracle
    model.hprod_available = has_hessian_operator && !has_oracle
    model.hess_available = has_hessian

    # Initialize evaluator using model's structure.
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if has_hessian_operator
        push!(init_feat, :HessVec)
    end
    if has_nlp_constraints
        push!(init_feat, :Jac)
    end
    if has_jacobian_operator
        push!(init_feat, :JacVec)
    end
    MOI.initialize(model.nlp_data.evaluator, init_feat)

    # Sparsity
    jacobian_sparsity = MOI.jacobian_structure(model)
    nnzj = length(jacobian_sparsity)
    jrows = Vector{Int}(undef, nnzj)
    jcols = Vector{Int}(undef, nnzj)
    for k in 1:nnzj
        jrows[k], jcols[k] = jacobian_sparsity[k]
    end
    model.jrows = jrows
    model.jcols = jcols

    hessian_sparsity = has_hessian ? MOI.hessian_lagrangian_structure(model) : Tuple{Int,Int}[]
    nnzh = length(hessian_sparsity)
    hrows = Vector{Int}(undef, nnzh)
    hcols = Vector{Int}(undef, nnzh)
    for k in 1:nnzh
        hrows[k], hcols[k] = hessian_sparsity[k]
    end
    model.hrows = hrows
    model.hcols = hcols

    model.needs_new_nlp = true
    return
end


@assert optim.solver isa Nothing
# MadNLPMOI._setup_model(optim)
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
end
@assert !optim.invalid_model
array_type = pop!(optim.options, :array_type, nothing)
# MadNLPMOI._setup_nlp(optim; array_type=nothing) # array_type comes from pop!(optim, :array_type, nothing)
begin
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


function _setup_nlp(model::Optimizer; array_type = nothing)
    if !model.needs_new_nlp
        return model.nlp
    end

    # Number of nonzeros for the jacobian and hessian of the Lagrangian
    nnzj = length(model.jrows)
    nnzh = length(model.hrows)

    # Initial variable
    nvar = length(model.variables.lower)
    x0 = zeros(Float64, nvar)
    for i in 1:length(model.variable_primal_start)
        x0[i] = if model.variable_primal_start[i] !== nothing
            model.variable_primal_start[i]
        else
            clamp(0.0, model.variables.lower[i], model.variables.upper[i])
        end
    end

    # Constraints bounds
    g_L, g_U = copy(model.qp_data.g_L), copy(model.qp_data.g_U)
    for (_, s) in model.vector_nonlinear_oracle_constraints
        append!(g_L, s.set.l)
        append!(g_U, s.set.u)
    end
    for bound in model.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    ncon = length(g_L)

    # Dual multipliers
    y0 = zeros(Float64, ncon)
    for (i, start) in enumerate(model.qp_data.mult_g)
        y0[i] = _dual_start(model, start, -1)
    end
    offset = length(model.qp_data.mult_g)
    if model.nlp_dual_start === nothing
        # First there is VectorNonlinearOracle...
        for (_, cache) in model.vector_nonlinear_oracle_constraints
            if cache.start !== nothing
                for i in 1:cache.set.output_dimension
                    y0[offset+i] = _dual_start(model, cache.start[i], -1)
                end
            end
            offset += cache.set.output_dimension
        end
        # ...then come the ScalarNonlinearFunctions
        for (key, val) in model.mult_g_nlp
            y0[offset+key.value] = _dual_start(model, val, -1)
        end
    else
        for (i, start) in enumerate(model.nlp_dual_start::Vector{Float64})
            y0[offset+i] = _dual_start(model, start, -1)
        end
    end

    nlp = MOIModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = model.variables.lower,
            uvar = model.variables.upper,
            ncon = ncon,
            y0 = y0,
            lcon = g_L,
            ucon = g_U,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = model.sense == MOI.MIN_SENSE,
            islp = model.islp,
            name = "MOIModel",
            variable_bounds_analysis=false,
            constraint_bounds_analysis=false,
            jprod_available = model.jprod_available,
            hprod_available = model.hprod_available,
            hess_available = model.hess_available,
        ),
        model,
        NLPModels.Counters(),
    )

    model.nlp = if isnothing(array_type)
        nlp
    else
        MadNLP.SparseWrapperModel(array_type, nlp)
    end

    model.needs_new_nlp = false
    return model.nlp
end

function MOI.optimize!(model::Optimizer)
    if model.solver === nothing
        _setup_model(model)
    end
    if model.invalid_model
        return
    end

    if model.nlp_model !== nothing
        empty!(model.qp_data.parameters)
        for (p, index) in model.parameters
            model.qp_data.parameters[p.value] = model.nlp_model[index]
        end
    end

    array_type = pop!(model.options, :array_type, nothing)
    _setup_nlp(model; array_type = array_type)

    if model.silent
        model.options[:print_level] = MadNLP.ERROR
    end
    options = copy(model.options)
    # Specific options depending on problem's structure.
    if !model.hess_available
        options[:hessian_approximation] = MadNLP.CompactLBFGS
    end
    # Set Jacobian to constant if all constraints are linear.
    if model.has_only_linear_constraints
        options[:jacobian_constant] = true
    end
    # Clear timers
    for (_, s) in model.vector_nonlinear_oracle_constraints
        s.eval_f_timer = 0.0
        s.eval_jacobian_timer = 0.0
        s.eval_hessian_lagrangian_timer = 0.0
    end
    # Instantiate MadNLP.
    model.solver = MadNLP.MadNLPSolver(model.nlp; options...)
    result = MadNLP.solve!(model.solver)
    model.result = if isnothing(array_type)
        result
    else
        model.options[:array_type] = array_type
        copy_result_to_cpu(result)
    end
    model.solve_time = model.solver.cnt.total_time
    model.solve_iterations = model.solver.cnt.k
    return
end

