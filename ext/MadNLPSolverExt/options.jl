export MadNLPOptions

@kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
    # Primary options
    tol::Float64 = 1e-8
    max_iter::Int = 1000
    print_level::Int = 3 # corresponds to MadNLP.LogLevels(print_level) where TRACE = 1 <= print_level <= 6 = ERROR
    eval_hessian::Bool = true

    ##

    # Primary options
    tol::T = 1e-8

    # Termination options
    s_max::T = 100.
    max_iter::Int = 3000
    max_wall_time::T = 1e6
    acceptable_tol::T = 1e-6
    acceptable_iter::Int = 15
    diverging_iterates_tol::T = 1e20

    # NLP options
    kappa_d::T = 1e-5
    # fixed_variable_treatment::Symbol = :make_parameter # [:make_parameter, :relax_bound]
    # # equality_treatment = ?
    bound_relax_factor::T = 1e-8
    jacobian_constant::Bool = false
    hessian_constant::Bool = false
    hessian_approximation::Symbol = :exact_hessian
    # # quasi_newton_options = ?
    # # inertia_correction_method = ?
    # # intertia_free_tol = ?
    # default_primal_regularization::T = 0.
    # default_dual_regularization::T = 0.

    # initialization options
    dual_initialized::Bool = false
    # dual_initialization_method::Symbol = :zero # [:zero, :least_squares]
    constr_mult_init_max::T = 1e3
    bound_push::T = 1e-2 # (corresponds to kappa_1)
    bound_fac::T = 1e-2 # should be "bound_frac" (corresponds to kappa_2)
    nlp_scaling::Bool = true # (corresponds to [:gradient => true, :none => false], with no support for :user, :equilibration)
    nlp_scaling_max_gradient::T = 100.

    # # Hessian Perturbation
    # min_hessian_perturbation::T = 1e-20
    # first_hessian_perturbation::T = 1e-4
    # max_hessian_perturbation::T = 1e20
    # perturb_inc_fact_first::T = 1e2
    # perturb_inc_fact::T = 8.
    # perturb_dec_fact::T = 1/3
    # jacobian_regularization_exponent::T = 1/4
    # jacobian_regularization_value::T = 1e-8

end
