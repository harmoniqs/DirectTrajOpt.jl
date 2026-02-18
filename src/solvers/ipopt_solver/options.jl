export IpoptOptions

"""
    Solver options for Ipopt

    https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_print_options_documentation
"""
Base.@kwdef mutable struct IpoptOptions <: Solvers.AbstractSolverOptions
    tol::Float64 = 1e-8
    s_max::Float64 = 100.0
    max_iter::Int = 1_000
    max_cpu_time = 1_000_000.0
    dual_inf_tol::Float64 = 1.0
    constr_viol_tol::Float64 = 1.0e-6
    compl_inf_tol::Float64 = 1.0e-3
    acceptable_tol::Float64 = 1.0e-6
    acceptable_iter::Int = 15
    acceptable_dual_inf_tol::Float64 = 1.0e10
    acceptable_constr_viol_tol::Float64 = 1.0e-2
    acceptable_compl_inf_tol::Float64 = 1.0e-2
    acceptable_obj_change_tol::Float64 = 1.0e-5
    diverging_iterates_tol::Float64 = 1.0e8
    eval_hessian = true
    hessian_approximation = eval_hessian ? "exact" : "limited-memory"
    hsllib = nothing
    inf_pr_output = "original"
    linear_solver = "mumps"
    mu_strategy = "adaptive"
    refine = true
    adaptive_mu_globalization = refine ? "obj-constr-filter" : "never-monotone-mode"
    mu_target::Float64 = 1.0e-4
    nlp_scaling_method = "gradient-based"
    output_file = nothing
    print_level::Int = 5
    print_user_options = "no"
    print_options_documentation = "no"
    print_timing_statistics = "no"
    print_options_mode = "text"
    print_advanced_options = "no"
    print_info_string = "no"
    print_frequency_iter = 1
    print_frequency_time = 0.0
    recalc_y = "no"
    recalc_y_feas_tol = 1.0e-6
    watchdog_shortened_iter_trigger = 0
    watchdog_trial_iter_max = 3
end
