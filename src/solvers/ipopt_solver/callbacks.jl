module Callbacks

using ..DirectTrajOpt
using Ipopt

"""
    # Callbacks evaluated by Ipopt should have the following signature:
    # Note that Cint === Int32 && Cdouble === Float64

    function my_intermediate_callback(
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint,
    )
        # ... user code ...
        return true # or `return false` to terminate the solve.
    end
"""

# Take 1

function callback_update_trajectory(problem::DirectTrajOptProblem; callback=nothing)
    function __callback(optimizer::Ipopt.Optimizer)
        function _callback(args...)
            IpoptSolverExt.update_trajectory!(prob, optimizer, optimizer.list_of_variable_indices)
            if callback isa Nothing
                return true
            end
            # by now, the trajectory is up to date, so `callback` can make use of it for e.g. rollouts
            return callback(args...)
        end
        return _callback
    end
    return __callback
end

function callback_update_trajectory_with_rollout(problem::DirectTrajOptProblem, fid_fn::Function; callback=nothing, fid_thresh=0.99, freq=1)
    function __callback(optimizer::Ipopt.Optimizer)
        function _callback(args...)
            IpoptSolverExt.update_trajectory!(prob, optimizer, optimizer.list_of_variable_indices)

            res = (callback isa Nothing) || (callback(args...))

            # we should evaluate `fid_fn` every `freq` iterations even if !res
            if args[2] % freq == 0
                res_fid = fid_fn(prob.trajectory) < fid_thresh
                return res && res_fid
            end

            return res
        end
        return _callback
    end
    return __callback
end

# Take 2

function callback_factory(callbacks...; kwargs...)
    function _callback_factory(optimizer::Ipopt.Optimizer)
        function _callback(optimizer_state...)
            for callback in callbacks
                res = callback(optimizer, optimizer_state; kwargs)
                if !res
                    return false
                end
            end
            return true
        end
    end
end

function _callback_say_hello_factory(msg)
    function _callback_say_hello(optimizer, optimizer_state; kwargs...)
        println(msg)
        return true
    end
end

function _callback_stop_iteration_factory(stop_iteration)
    function _callback_stop_iteration(optimizer, optimizer_state; kwargs...)
        if optimizer_state[2] >= stop_iteration
            return false
        end
        return true
    end
end

function _callback_update_trajectory_factory(problem)
    function _callback_update_trajectory(optimizer, optimizer_state; kwargs...)
        IpoptSolverExt.update_trajectory!(problem, optimizer, optimizer.list_of_variable_indices)
        return true
    end
end

# WARNING: This callback expects that _callback_update_trajectory was evaluated beforehand
#          However, a custom callback can just as well do both in one go, especially if the overhead from doing a trajectory update once per iteration is undesirable
function _callback_rollout_fidelity_factory(problem, system, fid_fn; fid_thresh=nothing, freq=1)
    function _callback_rollout_fidelity(optimizer, optimizer_state; kwargs...)
        if optimizer_state[2] % freq != 0
            return true
        end

        fid = fid_fn(problem.trajectory, system)

        # Probably comment this out and/or customize display of fidelities
        println()
        println("Fidelity: ", fid)

        return fid_thresh isa Nothing || fid < fid_thresh
    end
end

# Example usage:
#
# # The following solve should proceed as usual, printing the current fidelity (as computed by unitary_rollout_fidelity) once every 10 iterations, and stopping once it exceeds 0.999
# > initial = unitary_rollout_fidelity(prob.trajectory, sys)
# > cb = callback_factory(_callback_update_trajectory_factory(prob), _callback_rollout_fidelity_factory(prob, sys, unitary_rollout_fidelity; fid_thresh=0.999, freq=10))
# > solve!(prob; max_iter=100, callback=cb)
# > final = unitary_rollout_fidelity(prob.trajectory, sys)
# > @assert final > initial
# 
# # Terminating the solve manually (via Ctrl+C) will result in the final fidelity matching the initial fidelity (loss of solver progress) if _callback_update_trajectory is omitted
# > do_traj_update = false
# > initial = unitary_rollout_fidelity(prob.trajectory, sys)
# > cb = callback_factory((do_traj_update ? [_callback_update_trajectory_factory(prob)] : [])...)
# > solve!(prob; max_iter=100, callback=cb)
# > final = unitary_rollout_fidelity(prob.trajectory, sys)
# > @assert (final == initial) == (!do_traj_update)

