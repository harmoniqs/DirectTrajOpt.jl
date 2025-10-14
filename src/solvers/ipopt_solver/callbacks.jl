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
