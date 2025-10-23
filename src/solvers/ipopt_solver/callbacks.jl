module Callbacks


using ..DirectTrajOpt
using NamedTrajectories
using Ipopt

using TestItemRunner


# """
#     # Callbacks evaluated by Ipopt should have the following signature:
#     # Note that Cint === Int32 && Cdouble === Float64

#     function my_intermediate_callback(
#         alg_mod::Cint,
#         iter_count::Cint,
#         obj_value::Float64,
#         inf_pr::Float64,
#         inf_du::Float64,
#         mu::Float64,
#         d_norm::Float64,
#         regularization_size::Float64,
#         alpha_du::Float64,
#         alpha_pr::Float64,
#         ls_trials::Cint,
#     )
#         # ... user code ...
#         return true # or `return false` to terminate the solve.
#     end
# """

# """
# # Example usage:

# The following solve should proceed as usual, printing the current fidelity (as computed by unitary_rollout_fidelity) once every 10 iterations, and stopping once it exceeds 0.999
# > initial = unitary_rollout_fidelity(prob.trajectory, sys)
# > cb = callback_factory(_callback_update_trajectory_factory(prob), _callback_rollout_fidelity_factory(prob, sys, unitary_rollout_fidelity; fid_thresh=0.999, freq=10))
# > solve!(prob; max_iter=100, callback=cb)
# > final = unitary_rollout_fidelity(prob.trajectory, sys)
# > @assert final > initial

# Terminating the solve manually (via Ctrl+C) will result in the final fidelity matching the initial fidelity (loss of solver progress) if _callback_update_trajectory is omitted
# > do_traj_update = false
# > initial = unitary_rollout_fidelity(prob.trajectory, sys)
# > cb = callback_factory((do_traj_update ? [_callback_update_trajectory_factory(prob)] : [])...)
# > solve!(prob; max_iter=100, callback=cb)
# > final = unitary_rollout_fidelity(prob.trajectory, sys)
# > @assert (final == initial) == (!do_traj_update)
# """


"""
    IpoptOptimizerState

A shorthand referring to a NamedTuple of Int32 and Float64 inputs, which are forwarded to callbacks by Ipopt
"""

IpoptOptimizerState = NamedTuple{(:alg_mod, :iter_count, :obj_value, :inf_pr, :inf_du, :mu, :d_norm, :regularization_size, :alpha_du, :alpha_pr, :ls_trials), Tuple{Int32, Int32, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int32}}

"""
    callback_factory(callbacks...; kwargs...)

A factory method returning a single, unified callback, which may be passed to the `callback` kwarg of DirectTrajOpt.solve!.
The callbacks are executed in the order they are passed; once all callbacks have executed, the solver will continue to the next iteration if and only if every callback returned `true`.

# Argument list
- `callbacks...`: An optional variable-length tuple of arguments consisting of `Function`s with the following type signature: `function _callback_template(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)::Bool`
- `kwargs...`: An optional variable-length named tuple of arguments which are forwarded once to each callback

# Note
It is recommended that the first callback passed to this factory method be _callback_update_trajectory (returned by `callback_update_trajectory_factory`), as other callbacks often rely on the trajectory being kept up-to-date for e.g. storing trajectory histories, computing rollouts, etc.
"""

function callback_factory(callbacks...; kwargs...)
    function _callback_factory(optimizer::Ipopt.Optimizer)
        function _callback(optimizer_state...)
            return all([callback(optimizer, IpoptOptimizerState(optimizer_state); kwargs...) for callback in callbacks])
        end
    end
end

"""
    function callback_say_hello_factory(msg::String)

A simple callback factory returning a callback that prints a preselected message `msg` to stdout and then allows the solver to proceed
"""

function callback_say_hello_factory(msg::String)
    function _callback_say_hello(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        println(msg)
        return true
    end
end

"""
    function callback_stop_iteration_factory(stop_iteration::Int)

A simple callback factory returning a callback which stops the solver if it passes `stop_iteration` iterations; similar in effect to `solve!(...; max_iter=stop_iteration)`
"""

function callback_stop_iteration_factory(stop_iteration::Int)
    function _callback_stop_iteration(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        if optimizer_state.iter_count >= stop_iteration
            return false
        end
        return true
    end
end

"""
    function callback_update_trajectory_factory(problem::DirectTrajOptProblem)

A callback factory returning a callback that updates the `NamedTrajectory` associated with `problem`, using the optimizer's collection of stored primal variables
"""

function callback_update_trajectory_factory(problem::DirectTrajOptProblem)
    function _callback_update_trajectory(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        IpoptSolverExt.update_trajectory!(problem, optimizer, optimizer.list_of_variable_indices)
        return true
    end
end


"""
    function callback_update_trajectory_history_factory(problem::DirectTrajOptProblem, trajectories::Vector{<:NamedTrajectory})

A callback factory returning a callback that populates `trajectories` with a `deepcopy` of the `NamedTrajectory` associated with `problem` at each iteration.
Useful for debugging.

# Warning:
This callback expects that it be called after `_callback_update_trajectory`; if `_callback_update_trajectory` is not included alongside this one, `trajectories` will be populated with the same trajectory every time (typically undesirable).

# Todo:
Consider just storing the data field of each trajectory; we should not expect trajectory structure to change during a solve
"""

function callback_update_trajectory_history_factory(problem::DirectTrajOptProblem, trajectories::Vector{<:NamedTrajectory})
    function _callback_update_trajectory_history(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        push!(trajectories, deepcopy(problem.trajectory))
        return true
    end
end

"""
    function callback_rollout_fidelity_factory(problem::DirectTrajOptProblem, system::Any, fid_fn::Function; fid_thresh=nothing, freq=1)

A callback factory returning a callback that computes the rollout fidelity associated with an intermediate trajectory via `fid_fn(problem.trajectory, system)`, once every `freq` iterations, and stops the solver in its tracks if `!(fid_thresh isa Nothing) && fid >= fid_thresh`.
This is particularly useful for the early stages of a solve, when dynamics constraints are yet to be satisfied, during which time changes in the objective are a poor proxy for the true infidelity of the system at its final timestep.

# Warnings:
- This callback expects that it be called after `_callback_update_trajectory`
- This callback is meant to be used with `QuantumCollocation`, though it is not strictly necessary; a custom rollout method may be used in place of e.g. `QuantumCollocation.unitary_rollout_fidelity`, as long as it has the correct type signature
"""
function callback_rollout_fidelity_factory(problem::DirectTrajOptProblem, system::Any, fid_fn::Function; fid_thresh=nothing, freq=1)
    function _callback_rollout_fidelity(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        if optimizer_state.iter_count % freq != 0
            return true
        end

        fid = fid_fn(problem.trajectory, system)

        # Probably comment this out and/or customize display of fidelities
        println()
        println("Fidelity: ", fid)

        return fid_thresh isa Nothing || fid < fid_thresh
    end
end

"""
    function callback_best_rollout_fidelity_factory(problem::DirectTrajOptProblem, system::Any, fid_fn::Function, trajectories::Dict{Int32, Any}; fid_thresh=nothing, max_trajectories=1, freq=1)

A callback factory returning a callback similar to a combination of `_callback_update_trajectory_history` and `_callback_rollout_fidelity`, with two exceptions:
- `trajectories` is populated with a mapping from iteration index to trajectory, rather than being populated with an ordered list of all trajectories
- `trajectories` is populated with at most the `max_trajectories` best trajectories; poorer-performing trajectories (as measured by `fid_fn`) are then dropped
"""

function callback_best_rollout_fidelity_factory(problem::DirectTrajOptProblem, system::Any, fid_fn::Function, trajectories::Dict{Int32, Any}; fid_thresh=nothing, max_trajectories=1, freq=1)
    best_fid_idxs = Int32[]
    
    function _callback_best_rollout_fidelity(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        if optimizer_state.iter_count % freq != 0
            return true
        end

        fid = fid_fn(problem.trajectory, system)

        iter = optimizer_state.iter_count
        pushed_traj = false
        for i in 1:Int(min(length(best_fid_idxs), max_trajectories))
            if trajectories[best_fid_idxs[i]][1] < fid
                if length(best_fid_idxs) < max_trajectories
                    push!(best_fid_idxs, iter)
                    (best_fid_idxs[i], best_fid_idxs[i + 1:end]) = (best_fid_idxs[end], best_fid_idxs[i:end - 1])
                else
                    pop!(trajectories, best_fid_idxs[max_trajectories])
                    (best_fid_idxs[i], best_fid_idxs[i + 1:end]) = (iter, best_fid_idxs[i:end - 1])
                end

                push!(trajectories, Pair(iter, (fid, deepcopy(problem.trajectory))))

                pushed_traj = true
                break
            end
        end
        if !pushed_traj && length(best_fid_idxs) < max_trajectories
            push!(best_fid_idxs, iter)
            push!(trajectories, Pair(iter, (fid, deepcopy(problem.trajectory))))
        end

        # Probably comment this out and/or customize display of fidelities
        println()
        println("Fidelity: ", fid)
        # println("Best fidelity indices: ", best_fid_idxs)
        println("Best fidelities: ")
        for (k, (v, _)) in trajectories
            println(k, ": ", v)
        end

        return fid_thresh isa Nothing || fid < fid_thresh
    end
end

function cb_test()
    include("test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, traj, :x, :u),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(u -> [norm(u) - 1.0], :u, traj; times=2:traj.T-1, equality=false)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints=AbstractConstraint[g_u_norm])

    callback = callback_factory(
        callback_say_hello_factory("Hello, world!"),
        callback_stop_iteration_factory(50),
    )

    optimizer, variables = get_optimizer_and_variables(prob, IpoptOptions(; max_iter=100), callback)
    MOI.optimize!(optimizer)

    # solve!(prob; max_iter=100)
end

@testitem begin cb_test() end


end
