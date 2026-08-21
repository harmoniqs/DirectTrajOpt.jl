@testitem "Aqua quality assurance" tags=[:aqua] begin
    using Aqua, DirectTrajOpt

    Aqua.test_all(
        DirectTrajOpt;
        deps_compat = (check_extras = false,),
        # `hessian_structure` is exported by three different DirectTrajOpt
        # submodules (CommonInterface, Constraints, Integrators); the conflict
        # makes it appear undefined at DirectTrajOpt's surface even though all
        # three sub-definitions exist. TODO: pick a single canonical owner and
        # have the other submodules `import ..CommonInterface: hessian_structure`
        # rather than re-export.
        undefined_exports = (broken = true,),
        # persistent_tasks flakes on GitHub's 1.12 runners: it fails
        # intermittently on PR CI while passing locally on every platform
        # tried, and it has failed MAIN runs too (observed across #133,
        # #135, #136, and the 0.10.0 release PR — twice consecutively).
        # The check runs `has_persistent_tasks` against a freshly
        # precompiled package and is sensitive to runner-level async
        # timing (Julia 1.12.7's scheduler). Mark broken-with-evidence
        # rather than gating every release on a coin flip; revisit when
        # the Aqua/1.12 interaction is understood.
        persistent_tasks = (broken = true,),
    )
end
