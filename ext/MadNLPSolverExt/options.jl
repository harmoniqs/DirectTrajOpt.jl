export MadNLPOptions

# @kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
#     # Primary options
#     tol::Float64 = 1e-8
#     max_iter::Int = 1000
#     print_level::Int = 3 # corresponds to MadNLP.LogLevels(print_level) where TRACE = 1 <= print_level <= 6 = ERROR
#     eval_hessian::Bool = true


#     # tol::Float64 = 1e-8
#     # callback::String =  "sparse" # sparse | dense
#     # kkt_system::String = "reduced" # unreduced | reduced | condensed
#     # linear_solver::String = "mumps" # lapack | ldl | mumps

#     # print_level::Int = 3 # TRACE=1 | DEBUG=2 | INFO=3 | NOTICE=4 | WARN=5 | ERROR=6

#     # max_iter::Int = 3000

#     ##

#     # # Primary options
#     # tol::T = 1e-8
#     # callback::Symbol # [:sparse, :dense]
#     # kkt_system::Symbol # [:unreduced, :reduced, :condensed]
#     # linear_solver::Symbol # [:lapack, :ldl, :mumps]

#     # # General options
#     # rethrow_error::Bool = true
#     # disable_garbage_collector::Bool = false
#     # blas_num_threads::Int = 1
#     # # iterator::Type = RichardsonIterator
#     # # intermediate_callback::AbstractUserCallback = NoUserCallback()

#     # # Output options
#     # output_file::String = ""
#     # # print_level::LogLevels = INFO
#     # print_level::Int = 3
#     # # file_print_level::LogLevels = INFO
#     # file_print_level::Int = 3

#     # # Termination options
#     # acceptable_tol::T = 1e-6
#     # acceptable_iter::Int = 15
#     # diverging_iterates_tol::T = 1e20
#     # max_iter::Int = 3000
#     # max_wall_time::T = 1e6
#     # s_max::T = 100.

#     # # NLP options
#     # kappa_d::T = 1e-5
#     # fixed_variable_treatment::Symbol = ((callback == :sparse && kkt_system == :condensed) ? :relax_bound : :make_parameter)
#     # equality_treatment::Symbol = ((callback == :sparse && kkt_system == :condensed) ? :relax_equality : :enforce_equality)
#     # bound_relax_factor::T = 1e-8
#     # jacobian_constant::Bool = false
#     # hessian_constant::Bool = false
#     # hessian_approximation::Symbol = :exact_hessian # [:exact_hessian, :bfgs, :damped_bfgs, :compact_lbfgs] # Note that the options :bfgs and :damped_bfgs are incompatible with sparse KKT systems/sparse linear solvers
#     # inertia_correction_method::Symbol = :inertia_auto # [:inertia_auto, :inertia_based, :inertia_ignore, :inertia_free] # Mechanism for regularizing reduced Hessian such that its inertia is (n_primal, 0, n_dual); if not using an "inertia-revealing" solver, this defaults to :inertia_free; see https://github.com/MadNLP/MadNLP.jl/blob/master/src/IPM/inertiacorrector.jl
#     # inertia_free_tol::T = 0. # Only applicable when `inertia_correction_method == :inertia_free`; see https://github.com/MadNLP/MadNLP.jl/blob/master/src/IPM/solver.jl
#     # default_primal_regularization::T = 0. # Recently added option of indeterminate utility; has no analog in Ipopt; see https://github.com/MadNLP/MadNLP.jl/pull/573
#     # default_dual_regularization::T = 0. # Recently added option of indeterminate utility; has no analog in Ipopt; see https://github.com/MadNLP/MadNLP.jl/pull/573

#     # quasi_newton_init_strategy::Symbol = :scalar1 # [:scalar_1, :scalar_2, :scalar_3, :scalar_4, :constant]
#     # quasi_newton_max_history::Int = 6
#     # quasi_newton_init_value::T = 1.0
#     # quasi_newton_sigma_min::T = 1e-8
#     # quasi_newton_sigma_max::T = 1e+8   

#     # # Initialization options
#     # dual_initialized::Bool = false
#     # # dual_initialization_method::Symbol = :least_squares # [:least_squares, :set_to_zero]
#     # dual_initialization_method::Symbol = ((callback == :sparse && kkt_system == :condensed) ? :set_to_zero : :least_squares)
#     # constr_mult_init_max::T = 1e3
#     # bound_push::T = 1e-2 # (corresponds to kappa_1)
#     # bound_fac::T = 1e-2 # should be "bound_frac" (corresponds to kappa_2)
#     # nlp_scaling::Bool = true # (corresponds to [:gradient => true, :none => false], with no support for :user, :equilibration)
#     # nlp_scaling_max_gradient::T = 100.

#     # # Hessian Perturbation
#     # min_hessian_perturbation::T = 1e-20
#     # first_hessian_perturbation::T = 1e-4
#     # max_hessian_perturbation::T = 1e20
#     # perturb_inc_fact_first::T = 1e2
#     # perturb_inc_fact::T = 8.
#     # perturb_dec_fact::T = 1/3
#     # jacobian_regularization_exponent::T = 1/4
#     # jacobian_regularization_value::T = 1e-8

#     # # Restoration options
#     # soft_resto_pderror_reduction_factor::T = 0.9999 # soft restoration phase primal-dual error required reduction
#     # required_infeasibility_reduction::T = 0.9 # restoration phase termination criterion

#     # # Line search
#     # obj_max_inc::T = 5. # max acceptable order of magnitude increase in barrier objective
#     # kappa_soc::T = 0.99 # 2nd order correction sufficient reduction rule factor
#     # max_soc::Int = 4 # 2nd order correction max trial steps per iteration
#     # alpha_min_frac::T = 0.05 # minimal step size safety factor
#     # s_theta::T = 1.1 # switching rule current constraint violation exponent
#     # s_phi::T = 2.3 # switching rule linear barrrier function exponent
#     # eta_phi::T = 1e-4 # Armijo relaxation factor; Ipopt uses default of 1e-8
#     # gamma_theta::T = 1e-5 # constraint violation filter margin relaxation factor
#     # gamma_phi::T = 1e-5 # barrier function filter margin relaxation factor; Ipopt uses default of 1e-8
#     # delta::T = 1. # switching rule constraint violation multiplier
#     # kappa_sigma::T = 1e10 # primal estimate-dual variable deviation limitation factor
#     # barrier_tol_factor::T = 10. # barrier stop test mu factor
#     # rho::T = 1000. # restoration penalty parameter (corresponds to IpoptOptions.resto_penalty_parameter, NOT IpoptOptions.rho)
    
#     # # kappha_soc::T = 0.99 # typo of `kappa_soc`

#     # # Barrier
#     # barrier::Symbol = :monotone # [:monotone, :adaptive_quality_function, :adaptive_loqo] # barrier update algorithm
#     # tau_min::T = 0.99 # lower bound on fraction to boundary

#     # barrier_update_mu_init::T = 1e-1
#     # barrier_update_mu_min::T = 1e-11
#     # barrier_update_mu_superlinear_decrease_power::T = 1.5
#     # barrier_update_mu_linear_decrease_factor::T = .2
#     # barrier_update_mu_max::T = 1e5
#     # barrier_update_sigma_min::T = 1e-6
#     # barrier_update_sigma_max::T = 1e2
#     # barrier_update_sigma_tol::T = 1e-2
#     # barrier_update_gamma::T = ((barrier == :adaptive_loqo) ? 0.1 : 1.0)
#     # barrier_update_max_gs_iter::Int = 8
#     # barrier_update_free_mode::Bool = true
#     # barrier_update_globalization::Bool = true
#     # barrier_update_n_update::Int = 0
#     # barrier_update_r::T = .95

#     # # # Barrier update options (:monotone, :adaptive_quality_function, :adaptive_loqo)
#     # # barrier_update_mu_init::T = 1e-1
#     # # barrier_update_mu_min::T = 1e-11
#     # # barrier_update_mu_superlinear_decrease_power::T = 1.5
#     # # barrier_update_mu_linear_decrease_factor::T = .2

#     # # # Additional barrier update options (:adaptive_quality_function)
#     # # barrier_update_mu_max::T = 1e5
#     # # barrier_update_sigma_min::T = 1e-6
#     # # barrier_update_sigma_max::T = 1e2
#     # # barrier_update_sigma_tol::T = 1e-2
#     # # barrier_update_gamma::T = 1.0
#     # # barrier_update_max_gs_iter::Int = 8
#     # # barrier_update_free_mode::Bool = true
#     # # barrier_update_globalization::Bool = true
#     # # barrier_update_n_update::Int = 0

#     # # # Additional barrier update options (:adaptive_loqo)
#     # # mu_init::T = 1e-1
#     # # mu_min::T = 1e-11
#     # # mu_max::T = 1e5
#     # # gamma::T = 0.1 # scale factor
#     # # r::T = .95 # Steplength param
#     # # mu_superlinear_decrease_power::T = 1.5
#     # # mu_linear_decrease_factor::T = .2
#     # # free_mode::Bool = true
#     # # globalization::Bool = true
# end











# # function parse_options(options::OptionStruct)
# #     opts = Dict{Symbol, Any}()
# #     pushopt(k::Symbol, v::Any) = push!(opts, Pair(k, v))
# #     pushopt(k::Symbol, v::Nothing) = @assert false "failed to set value of $(k)"

# #     for (k, t) = zip(fieldnames(typeof(options)), fieldtypes(typeof(options)))
# #         v = getfield(options, k)

# #         if !isa(t, Type{String})
# #             pushopt(k, v)
# #         else
# #             if k == :opt
# #                 pushopt(k, get((_subopt1=_type1, _subopt2=_type2,), v, nothing))
# #             end
# #         end
# #     end

# #     return opts
# # end

# # function parse_options(options::OptionStruct)
# #     opts = Dict{Symbol, Any}()
# #     pushopt(k::Symbol, v::Any) = push!(opts, Pair(k, v))
# #     pushopt(k::Symbol, v::Nothing) = @assert false "failed to set value of $(k)"

# #     for (k, t) = zip(fieldnames(typeof(options)), fieldtypes(typeof(options)))
# #         v = getfield(options, k)

# #         if !isa(t, Type{String})
# #             pushopt(k, v)
# #         else
# #             if k == :opt
# #                 pushopt(k, get((_subopt1=_type1, _subopt2=_type2,), v, nothing))
# #             end
# #         end
# #     end

# #     return opts
# # end



@kwdef mutable struct MadNLPOptions{T <: Float64}
    tol::Float64 = 1e-8
    callback::String =  "sparse" # sparse | dense
    kkt_system::String = "reduced" # unreduced | reduced | condensed
    linear_solver::String = "mumps" # lapack | ldl | mumps

    # General options
    rethrow_error::Bool = true
    disable_garbage_collector::Bool = false
    blas_num_threads::Int = 1
    iterator::String = "richardson_iterator"
    # intermediate_callback::MadNLP.AbstractUserCallback = MadNLP.NoUserCallback()

    # Output options
    output_file::String = ""
    print_level::Int = 3 # TRACE=1 | DEBUG=2 | INFO=3 | NOTICE=4 | WARN=5 | ERROR=6
    file_print_level::Int = 3 # TRACE=1 | DEBUG=2 | INFO=3 | NOTICE=4 | WARN=5 | ERROR=6

    # Termination options
    acceptable_tol::T = 1e-6
    acceptable_iter::Int = 15
    diverging_iterates_tol::T = 1e20
    max_iter::Int = 3000
    max_wall_time::T = 1e6
    s_max::T = 100.

    # NLP options
    kappa_d::T = 1e-5
    fixed_variable_treatment::String = ((callback == "sparse" && kkt_system == "condensed") ? "relax_bound" : "make_parameter") # "make_parameter" | "relax_bound"
    equality_treatment::String = ((callback == "sparse" && kkt_system == "condensed") ? "relax_equality" : "enforce_equality") # "enforce_equality" | "relax_equality"
    bound_relax_factor::T = 1e-8
    jacobian_constant::Bool = false
    hessian_constant::Bool = false
    hessian_approximation::String = "exact_hessian" # "exact_hessian" | "bfgs" | "damped_bfgs" | "compact_lbfgs" # Note that the options :bfgs and :damped_bfgs are incompatible with sparse KKT systems/sparse linear solvers
    inertia_correction_method::String = "inertia_auto" # "inertia_auto" | "inertia_based" | "inertia_ignore" | "inertia_free" # Mechanism for regularizing reduced Hessian such that its inertia is (n_primal, 0, n_dual); if not using an "inertia-revealing" solver, this defaults to :inertia_free; see https://github.com/MadNLP/MadNLP.jl/blob/master/src/IPM/inertiacorrector.jl
    inertia_free_tol::T = 0. # Only applicable when `inertia_correction_method == :inertia_free`; see https://github.com/MadNLP/MadNLP.jl/blob/master/src/IPM/solver.jl
    default_primal_regularization::T = 0. # Recently added option of indeterminate utility; has no analog in Ipopt; see https://github.com/MadNLP/MadNLP.jl/pull/573
    default_dual_regularization::T = 0. # Recently added option of indeterminate utility; has no analog in Ipopt; see https://github.com/MadNLP/MadNLP.jl/pull/573

    quasi_newton_init_strategy::String = "scalar_1" # "scalar_1" | "scalar_2" | "scalar_3" | "scalar_4" | "constant"
    quasi_newton_max_history::Int = 6
    quasi_newton_init_value::T = 1.0
    quasi_newton_sigma_min = 1.0e-8
    quasi_newton_sigma_max = 1.0e+8

    # Initialization options
    dual_initialized::Bool = false
    dual_initialization_method::Symbol = ((callback == :sparse && kkt_system == :condensed) ? :set_to_zero : :least_squares)
    constr_mult_init_max::T = 1e3
    bound_push::T = 1e-2 # (corresponds to kappa_1)
    bound_fac::T = 1e-2 # should be "bound_frac" (corresponds to kappa_2)
    nlp_scaling::Bool = true # (corresponds to [:gradient => true, :none => false], with no support for :user, :equilibration)
    nlp_scaling_max_gradient::T = 100.

    # Hessian Perturbation
    min_hessian_perturbation::T = 1e-20
    first_hessian_perturbation::T = 1e-4
    max_hessian_perturbation::T = 1e20
    perturb_inc_fact_first::T = 1e2
    perturb_inc_fact::T = 8.
    perturb_dec_fact::T = 1/3
    jacobian_regularization_exponent::T = 1/4
    jacobian_regularization_value::T = 1e-8

    # Restoration options
    soft_resto_pderror_reduction_factor::T = 0.9999 # soft restoration phase primal-dual error required reduction
    required_infeasibility_reduction::T = 0.9 # restoration phase termination criterion

    # Line search
    obj_max_inc::T = 5. # max acceptable order of magnitude increase in barrier objective
    kappa_soc::T = 0.99 # 2nd order correction sufficient reduction rule factor
    max_soc::Int = 4 # 2nd order correction max trial steps per iteration
    alpha_min_frac::T = 0.05 # minimal step size safety factor
    s_theta::T = 1.1 # switching rule current constraint violation exponent
    s_phi::T = 2.3 # switching rule linear barrrier function exponent
    eta_phi::T = 1e-4 # Armijo relaxation factor; Ipopt uses default of 1e-8
    gamma_theta::T = 1e-5 # constraint violation filter margin relaxation factor
    gamma_phi::T = 1e-5 # barrier function filter margin relaxation factor; Ipopt uses default of 1e-8
    delta::T = 1. # switching rule constraint violation multiplier
    kappa_sigma::T = 1e10 # primal estimate-dual variable deviation limitation factor
    barrier_tol_factor::T = 10. # barrier stop test mu factor
    rho::T = 1000. # restoration penalty parameter (corresponds to IpoptOptions.resto_penalty_parameter, NOT IpoptOptions.rho)
    
    # Barrier
    barrier::String = "monotone" # "monotone" | "adaptive_quality_function" | "adaptive_loqo" # barrier update algorithm
    tau_min::T = 0.99 # lower bound on fraction to boundary

    barrier_update_mu_init::T = 1e-1
    barrier_update_mu_min::T = 1e-11
    barrier_update_mu_superlinear_decrease_power::T = 1.5
    barrier_update_mu_linear_decrease_factor::T = .2
    barrier_update_mu_max::T = 1e5
    barrier_update_sigma_min::T = 1e-6
    barrier_update_sigma_max::T = 1e2
    barrier_update_sigma_tol::T = 1e-2
    barrier_update_gamma::T = ((barrier == "adaptive_loqo") ? 0.1 : 1.0)
    barrier_update_max_gs_iter::Int = 8
    barrier_update_free_mode::Bool = true
    barrier_update_globalization::Bool = true
    barrier_update_n_update::Int = 0
    barrier_update_r::T = .95

    # eval_hessian::Bool = true
end


function parse_options(options::MadNLPOptions)
    opts = Dict{Symbol, Any}()
    quasi_newton_opts = Dict{Symbol, Any}()
    barrier_update_opts = Dict{Symbol, Any}()
    pushopt(k::Symbol, v::Any; d=opts) = push!(d, Pair(k, v))
    pushopt(k::Symbol, v::Nothing; d=opts) = @assert false "failed to set value of $(k)"

    for (k, t) = zip(fieldnames(typeof(options)), fieldtypes(typeof(options)))
        v = getfield(options, k)

        if length(String(k)) > length("quasi_newton_") && String(k)[1:length("quasi_newton_")] == "quasi_newton_"
            k = Symbol(String(k)[1 + length(String(:quasi_newton_)):end])
            if !isa(t, Type{String})
                pushopt(k, v; d=quasi_newton_opts)
            else
                v = Symbol(v)
                pushopt(k, get((scalar_1=MadNLP.SCALAR1, scalar_2=MadNLP.SCALAR2, scalar_3=MadNLP.SCALAR3, scalar_4=MadNLP.SCALAR4, constant=MadNLP.CONSTANT), v, nothing); d=quasi_newton_opts)
            end
        elseif length(String(k)) > length("barrier_update_") && String(k)[1:length("barrier_update_")] == "barrier_update_"
            k = Symbol(String(k)[1 + length(String(:barrier_update_)):end])
            pushopt(k, v; d=barrier_update_opts)
        elseif k == :print_level || k == :file_print_level
            pushopt(k, MadNLP.LogLevels(v))
        elseif !isa(t, Type{String})
            pushopt(k, v)
        else
            v = Symbol(v)

            # Primary options
            if k == :callback
                pushopt(k, get((sparse=MadNLP.SparseCallback,), v, nothing))
            elseif k == :kkt_system
                pushopt(k, get((unreduced=MadNLP.SparseUnreducedKKTSystem, reduced=MadNLP.SparseKKTSystem, condensed=MadNLP.SparseCondensedKKTSystem), v, nothing))
            elseif k == :linear_solver
                pushopt(k, get((mumps=MadNLP.MumpsSolver, lapack_cpu=MadNLP.LapackCPUSolver), v, nothing))
            end

            # General options
            if k == :iterator
                pushopt(k, get((richardson_iterator=MadNLP.RichardsonIterator,), v, nothing))
            end

            # NLP Options
            if k == :fixed_variable_treatment
                pushopt(k, get((make_parameter=MadNLP.MakeParameter, relax_bound=MadNLP.RelaxBound,), v, nothing))
            elseif k == :equality_treatment
                pushopt(k, get((enforce_equality=MadNLP.EnforceEquality, relax_equality=MadNLP.RelaxEquality,), v, nothing))
            elseif k == :hessian_approximation
                pushopt(k, get((exact_hessian=MadNLP.ExactHessian, bfgs=MadNLP.BFGS, damped_bfgs=MadNLP.DampedBFGS, compact_lbfgs=MadNLP.CompactLBFGS,), v, nothing))
            elseif k == :inertia_correction_method
                pushopt(k, get((inertia_auto=MadNLP.InertiaAuto, inertia_based=MadNLP.InertiaBased, inertia_ignore=MadNLP.InertiaIgnore, inertia_free=MadNLP.InertiaFree,), v, nothing))
            end

            # Initialization options
            if k == :dual_initialization_method
                pushopt(k, get((least_squares=MadNLP.DualInitializeLeastSquares, set_to_zero=MadNLP.DualInitializeSetZero), v, nothing))
            end

            # Barrier update algorithm
            if k == :barrier
                v = Symbol(v)
                pushopt(k, get((monotone=MadNLP.MonotoneUpdate, adaptive_quality_function=MadNLP.QualityFunctionUpdate, adaptive_loqo=MadNLP.LOQOUpdate), v, nothing))
            end
        end
    end

    quasi_newton_opts_final = MadNLP.QuasiNewtonOptions(
        init_strategy=quasi_newton_opts[:init_strategy],
        init_value=quasi_newton_opts[:init_value],
        max_history=quasi_newton_opts[:max_history],
        sigma_max=quasi_newton_opts[:sigma_max],
        sigma_min=quasi_newton_opts[:sigma_min],
    )

    barrier_update_opts_final = (opts[:barrier] == MadNLP.MonotoneUpdate) ? (MadNLP.MonotoneUpdate(
        mu_init=barrier_update_opts[:mu_init],
        mu_min=barrier_update_opts[:mu_min],
        mu_superlinear_decrease_power=barrier_update_opts[:mu_superlinear_decrease_power],
        mu_linear_decrease_factor=barrier_update_opts[:mu_linear_decrease_factor],
    )) : ((opts[:barrier] == MadNLP.QualityFunctionUpdate) ? (MadNLP.QualityFunctionUpdate(
        mu_init=barrier_update_opts[:mu_init],
        mu_min=barrier_update_opts[:mu_min],
        mu_max=barrier_update_opts[:mu_max],
        sigma_min=barrier_update_opts[:sigma_min],
        sigma_max=barrier_update_opts[:sigma_max],
        sigma_tol=barrier_update_opts[:sigma_tol],
        gamma=barrier_update_opts[:gamma],
        max_gs_iter=barrier_update_opts[:max_gs_iter],
        mu_superlinear_decrease_power=barrier_update_opts[:mu_superlinear_decrease_power],
        mu_linear_decrease_factor=barrier_update_opts[:mu_linear_decrease_factor],
        free_mode=barrier_update_opts[:free_mode],
        globalization=barrier_update_opts[:globalization],
        n_update=barrier_update_opts[:n_update],
    )) : MadNLP.LOQOUpdate(
        mu_init=barrier_update_opts[:mu_init],
        mu_min=barrier_update_opts[:mu_min],
        mu_max=barrier_update_opts[:mu_max],
        gamma=barrier_update_opts[:gamma],
        r=barrier_update_opts[:r],
        mu_superlinear_decrease_power=barrier_update_opts[:mu_superlinear_decrease_power],
        mu_linear_decrease_factor=barrier_update_opts[:mu_linear_decrease_factor],
        free_mode=barrier_update_opts[:free_mode],
        globalization=barrier_update_opts[:globalization],
    ))

    opts[:quasi_newton_options] = quasi_newton_opts_final
    opts[:barrier] = barrier_update_opts_final
    return opts

    # return opts, quasi_newton_opts_final, barrier_update_opts_final
end

