import MathOptInterface as MOI
import Ipopt, MadNLP
using DirectTrajOpt

include("DirectTrajOpt.jl/test/solver_test_utils.jl")

const MadNLPMOI = [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPMOI][1]
const MadNLPSolverExt = [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPSolverExt][1]


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


function preprocess(model::MadNLPMOI.Optimizer)
    @assert model.solver isa Nothing "`model.solver` initialized prematurely"
    MadNLPMOI._setup_model(model)
    
    @assert model.nlp_model isa Nothing "`model.nlp_model` backend initialized erroneously"
    array_type = pop!(model.options, :array_type, nothing)

    @assert model.needs_new_nlp "`model.nlp` backend initialized prematurely"
    MadNLPMOI._setup_nlp(model; array_type=array_type)

    @assert !model.needs_new_nlp "`model.nlp` backend failed to initialize"
    model.options[:print_level] = (model.silent ? MadNLP.ERROR : model.options[:print_level])
    model.options[:hessian_approximation] = (!model.hess_available ? MadNLP.CompactLBFGS : model.options[:hessian_approximation])
    model.options[:jacobian_constant] = (model.has_only_linear_constraints ? true : model.options[:jacobian_constant])

    for (_, s) = model.vector_nonlinear_oracle_constraints
        s.eval_f_timer = 0.
        s.eval_jacobian_timer = 0.
        s.eval_hessian_lagrangian_timer = 0.
    end

    model.solver = MadNLP.MadNLPSolver(model.nlp; model.options...)
    return model
end


prob = get_seeded_prob(42)
optim, vars = MadNLPSolverExt.get_optimizer_and_variables(prob, MadNLPSolverExt.MadNLPOptions(; max_iter=100), nothing)
optim.options[:hessian_approximation] = MadNLP.ExactHessian
optim.options[:jacobian_constant] = false
model = preprocess(optim)



##


mutable struct Optimizer <: MOI.AbstractOptimizer
    solver::Union{Nothing, MadNLP.MadNLPSolver} = nothing, # 
    nlp::Union{Nothing, MadNLP.NLPModels.AbstractNLPModel} = nothing, # 
    result::Union{Nothing, MadNLP.MadNLPExecutionStats{Float64}} = nothing, # 

    name::String = "",
    invalid_model::Bool = false, # 
    silent::Bool = false,
    options::Dict{Symbol,Any} = Dict{Symbol, Any}(),
    solve_time::Float64 = NaN, # 
    solve_iterations::Int = 0, # 
    sense::MOI.OptimizationSense = MOI.FEASIBILITY_SENSE, # 

    parameters::Dict{MOI.VariableIndex, MOI.Nonlinear.ParameterIndex} = Dict{MOI.VariableIndex, Float64}(), # 
    variables::MOI.Utilities.VariablesContainer{Float64} = MOI.Utilities.VariablesContainer{Float64}(),
    list_of_variable_indices::Vector{MOI.VariableIndex} = MOI.VariableIndex[], # 
    variable_primal_start::Vector{Union{Nothing,Float64}} = Union{Nothing, Float64}[], # 

    nlp_data::MOI.NLPBlockData = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false), # 
    nlp_dual_start::Union{Nothing, Vector{Float64}} = nothing, # 
    mult_g_nlp::Dict{MOI.Nonlinear.ConstraintIndex, Float64} = Dict{MOI.Nonlinear.ConstraintIndex, Float64}(), # 

    qp_data::QPBlockData{Float64} = QPBlockData{Float64}(), # 
    nlp_model::Union{Nothing,MOI.Nonlinear.Model} = nothing, # 
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation = MOI.Nonlinear.SparseReverseMode(),
    vector_nonlinear_oracle_constraints::Vector{Tuple{MOI.VectorOfVariables, _VectorNonlinearOracleCache}} = Tuple{MOI.VectorOfVariables, _VectorNonlinearOracleCache}[], # 

    jrows::Vector{Int} = Int[], # 
    jcols::Vector{Int} = Int[], # 
    hrows::Vector{Int} = Int[], # 
    hcols::Vector{Int} = Int[], # 
    needs_new_nlp::Bool = true, # 
    has_only_linear_constraints::Bool = false, # 
    islp::Bool = false, # 
    jprod_available::Bool = false, # 
    hprod_available::Bool = false, # 
    hess_available::Bool = false, # 
end

# QPBlockData
# _VectorNonlinearOracleCache

# `get_optimizer_and_variables`
#  - instantiates `MadNLP.Optimizer()`
#  - sets `MOI.set(optimizer, MOI.NLPBlock(), block_data)`
#    - ensuring `optimizer.nlp_data` is set and `optimizer.solver isa Nothing`
#  - sets `MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)`
#    - ensuring `optimizer.sense` is set and `optimizer.needs_new_nlp` is true
#  - calls `set_variables!(optimizer, prob.trajectory)`
#    - wherein `variables = MOI.add_variables(optimizer, (traj.dim * traj.N) + traj.global_dim)`
#    - sets `MOI.set(optimizer, MOI.VariablePrimalStart(), variables[1:(traj.dim * traj.N)], collect(traj.datavec))`
#    - sets `MOI.set(optimizer, MOI.VariablePrimalStart(), variables[((traj.dim * traj.N) + 1):((traj.dim * traj.N) + traj.global_dim)]))` 
#    - in general `MOI.set(optimizer, attr::MOI.VariablePrimalStart(), vi::MOI.VariableIndex(::Int64), value::Union{Nothing, Real})`
#      - ensuring `!_is_parameter(vi)` and `model.variable_primal_start[column(vi)] = value` and `model.needs_new_nlp = true`
#  - also available are MOI.set(model::Optimizer, ::MOI.NLPBlockDualStart, values::Union{Nothing, Vector})
#    - ensuring `model.nlp_dual_start = values` and `model.needs_new_nlp = true`
#  - note that `MOI.set(::Optimizer, MOI.ObjectiveFunction{F}, ::F) where F<:Union{MOI.VariableIndex, MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}}` does
#    - ensure `MOI.set(model.qp_data, attr, func)` and `model.solver = nothing`
#  - MOI.eval_objective:
#    - if model.sense != MOI.FEASIBILITY_SENSE && model.nlp_data.has_objective returns MOI.eval_objective(model.nlp_data.evaluator, x)


function preoptimize(model::MadNLPMOI.Optimizer)
    @assert model.solver isa Nothing "`model.solver` initialized prematurely"
    MadNLPMOI._setup_model(model)
    
    @assert model.nlp_model isa Nothing "`model.nlp_model` backend initialized erroneously"
    array_type = pop!(model.options, :array_type, nothing)

    @assert model.needs_new_nlp "`model.nlp` backend initialized prematurely"
    MadNLPMOI._setup_nlp(model; array_type=array_type)

    @assert !model.needs_new_nlp "`model.nlp` backend failed to initialize"
    model.options[:print_level] = (model.silent ? MadNLP.ERROR : model.options[:print_level])
    model.options[:hessian_approximation] = (!model.hess_available ? MadNLP.CompactLBFGS : model.options[:hessian_approximation])
    model.options[:jacobian_constant] = (model.has_only_linear_constraints ? true : model.options[:jacobian_constant])

    for (_, s) = model.vector_nonlinear_oracle_constraints
        s.eval_f_timer = 0.
        s.eval_jacobian_timer = 0.
        s.eval_hessian_lagrangian_timer = 0.
    end

    model.solver = MadNLP.MadNLPSolver(model.nlp; model.options...)
    
    model.result = MadNLP.solve!(model.solver)

    model.options[:array_type] = array_type
    model.result = (isa(array_type, Nothing) ? model.result : copy_result_to_cpu(model.result))
    model.solve_time = model.solver.cnt.total_time
    model.solve_iterations = model.solver.cnt.k
end

function solve(model::MadNLPMOI.Optimizer)
    model.result = MadNLP.solve!(model.solver)

    return model
end

function postprocess(model::MadNLPMOI.Optimizer; array_type=nothing)
    model.options[:array_type] = array_type
    model.result = (isa(array_type, Nothing) ? model.result : copy_result_to_cpu(model.result))
    model.solve_time = model.solver.cnt.total_time
    model.solve_iterations = model.solver.cnt.k

    return model
end

