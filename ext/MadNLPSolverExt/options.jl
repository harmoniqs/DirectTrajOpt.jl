export MadNLPOptions

@kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
    # Primary options
    tol::Float64 = 1e-8
    max_iter::Int = 1000
    print_level::Int = 3 # corresponds to MadNLP.LogLevels(print_level) where TRACE = 1 <= print_level <= 6 = ERROR
    eval_hessian::Bool = true

    # ##

    # # Primary options
    # tol::T = 1e-8

    # # TODO: add remaining primary, general, and output options

    # # Termination options
    # acceptable_tol::T = 1e-6
    # acceptable_iter::Int = 15
    # diverging_iterates_tol::T = 1e20
    # max_iter::Int = 3000
    # max_wall_time::T = 1e6
    # s_max::T = 100.

    # # NLP options
    # kappa_d::T = 1e-5
    # # fixed_variable_treatment::Symbol = :make_parameter # [:make_parameter, :relax_bound] # TODO: use given defaults as defaults UNLESS `kkt_system` type parameter is a MadNLP.SparseCondensedKKTSystem
    # # equality_treatment::Symbol = :enforce_equality # [:enforce_equality, :relax_equality] # TODO: use given defaults as defaults UNLESS `kkt_system` type parameter is a MadNLP.SparseCondensedKKTSystem
    # bound_relax_factor::T = 1e-8
    # jacobian_constant::Bool = false
    # hessian_constant::Bool = false
    # # hessian_approximation::Symbol = :exact_hessian # # TODO
    # # quasi_newton_options::Nothing = nothing # # TODO: at minimum, support setting BGFSInitStrategy
    # # inertia_correction_method::Symbol = :inertia_auto # TODO: what options are there?
    # # inertia_free_tol::T = 0. # TODO: should this default remain fixed given an alternative choice of `inertia_correction_method`?
    # default_primal_regularization::T = 0. # Recently added option of indeterminate utility; see https://github.com/MadNLP/MadNLP.jl/pull/573
    # default_dual_regularization::T = 0. # Recently added option of indeterminate utility; see https://github.com/MadNLP/MadNLP.jl/pull/573

    # # initialization options
    # dual_initialized::Bool = false
    # # dual_initialization_method::Symbol = :zero # [:zero, :least_squares] # TODO: ?
    # constr_mult_init_max::T = 1e3
    # bound_push::T = 1e-2 # (corresponds to kappa_1)
    # bound_fac::T = 1e-2 # should be "bound_frac" (corresponds to kappa_2)
    # nlp_scaling::Bool = true # (corresponds to [:gradient => true, :none => false], with no support for :user, :equilibration)
    # nlp_scaling_max_gradient::T = 100.

    # # Hessian Perturbation
    # min_hessian_perturbation::T = 1e-20
    # first_hessian_perturbation::T = 1e-4
    # max_hessian_perturbation::T = 1e20
    # perturb_inc_fact_first::T = 1e2
    # perturb_inc_fact::T = 8.
    # perturb_dec_fact::T = 1/3
    # jacobian_regularization_exponent::T = 1/4
    # jacobian_regularization_value::T = 1e-8

    # # restoration options
    # soft_resto_pderror_reduction_factor::T = 0.9999
    # required_infeasibility_reduction::T = 0.9

    # # Line search
    # # obj_max_inc::T = 5.
    # # kappha_soc::T = 0.99
    # max_soc::Int = 4
    # alpha_min_frac::T = 0.05
    # s_theta::T = 1.1
    # s_phi::T = 2.3
    # eta_phi::T = 1e-4 # Armijo relaxation factor; Ipopt uses default of 1e-8
    # kappa_soc::T = 0.99
    # gamma_theta::T = 1e-5 # Constraint violation filter margin relaxation factor
    # gamma_phi::T = 1e-5 # Barrier fn filter margin relaxation factor; Ipopt uses default of 1e-8
    # delta::T = 1 # Switching rule constraint violation multiplier
    # kappa_sigma::T = 1e10 # primal estimate-dual variable deviation limitation factor
    # barrier_tol_factor::T = 10. # Barrier stop test mu factor
    # rho::T = 1000. # Restoration penalty parameter (corresponds to IpoptOptions.resto_penalty_parameter, NOT IpoptOptions.rho)

    # # Barrier
    # # mu_min by courtesy of Ipopt
    # # barrier::AbstractBarrierUpdate{T} = MonotoneUpdate(T(tol), barrier_tol_factor) # supports MonotoneUpdate, and QualityFunctionUpdate <: AbstractAdaptiveUpdate, LOQOUpdate <: AbstractAdaptiveUpdate, each of which have further parameterizations.
    # # barrier::Symbol = :monotone # [:monotone, :adaptive] # TODO: add associated parameters common to all barrier update schemes as well as those particular to each adaptive scheme
    # tau_min::T = 0.99 # Lower bound on fraction to boundary

end


# struct ExactHessian{T, VT} <: AbstractHessian{T, VT} end

# struct BFGS{T, VT <: AbstractVector{T}} <: AbstractQuasiNewton{T, VT}
#     init_strategy::BFGSInitStrategy
#     is_instantiated::Base.RefValue{Bool}
#     sk::VT
#     yk::VT
#     bsk::VT
#     last_g::VT
#     last_x::VT
#     last_jv::VT
# end

# struct DampedBFGS{T, VT <: AbstractVector{T}} <: AbstractQuasiNewton{T, VT}
#     init_strategy::BFGSInitStrategy
#     is_instantiated::Base.RefValue{Bool}
#     sk::VT
#     yk::VT
#     bsk::VT
#     rk::VT
#     last_g::VT
#     last_x::VT
#     last_jv::VT
# end

# mutable struct CompactLBFGS{T, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}, B} <: AbstractQuasiNewton{T, VT}
#     init_strategy::BFGSInitStrategy
#     sk::VT
#     yk::VT
#     last_g::VT
#     last_x::VT
#     last_jv::VT
#     init_value::T
#     sigma_min::T
#     sigma_max::T
#     max_mem::Int
#     current_mem::Int
#     skipped_iter::Int
#     Sk::MT       # n x p
#     Yk::MT       # n x p
#     Lk::MT       # p x p
#     Mk::MT       # p x p (for Cholesky factorization Mₖ = Jₖ Jₖᵀ)
#     Tk::MT       # 2p x 2p
#     DkLk::MT     # p x p
#     U::MT        # n x p
#     V::MT        # n x p
#     E::MT        # (n+m) x 2p
#     H::MT        # (n+m) x 2p
#     Dk::VT       # p
#     _w1::VT
#     _w2::VT
#     additional_buffers::B
#     max_mem_reached::Bool
# end


# @kwdef mutable struct QuasiNewtonOptions{T} <: AbstractOptions
#     init_strategy::BFGSInitStrategy = SCALAR1
#     max_history::Int = 6
#     init_value::T = 1.0
#     sigma_min::T = 1e-8
#     sigma_max::T = 1e+8
# end

# @kwdef mutable struct MonotoneUpdate{T} <: AbstractBarrierUpdate{T}
#     mu_init::T = 1e-1
#     mu_min::T = 1e-11
#     mu_superlinear_decrease_power::T = 1.5
#     mu_linear_decrease_factor::T = .2
# end

# @kwdef mutable struct QualityFunctionUpdate{T} <: AbstractAdaptiveUpdate{T}
#     mu_init::T = 1e-1
#     mu_min::T = 1e-11
#     mu_max::T = 1e5
#     sigma_min::T = 1e-6
#     sigma_max::T = 1e2
#     sigma_tol::T = 1e-2
#     gamma::T = 1.0
#     max_gs_iter::Int = 8
#     # For non-free mode
#     mu_superlinear_decrease_power::T = 1.5
#     mu_linear_decrease_factor::T = .2
#     free_mode::Bool = true
#     globalization::Bool = true
#     n_update::Int = 0
# end

# @kwdef mutable struct LOQOUpdate{T} <: AbstractAdaptiveUpdate{T}
#     mu_init::T = 1e-1
#     mu_min::T = 1e-11
#     mu_max::T = 1e5
#     gamma::T = 0.1 # scale factor
#     r::T = .95 # Steplength param
#     mu_superlinear_decrease_power::T = 1.5
#     mu_linear_decrease_factor::T = .2
#     free_mode::Bool = true
#     globalization::Bool = true
# end
