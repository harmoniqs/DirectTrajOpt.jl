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
        # intermittently on 1.12 PR CI while passing locally on every
        # platform tried, and it has failed 1.12 MAIN runs too (observed
        # across #133, #135, #136, and the 0.10.0 release PR). On
        # 1.10/1.11 it passes deterministically — so the check is LIVE
        # there and broken only where it flakes (an unconditional
        # broken=true makes Aqua report 'Unexpected Pass' as an error on
        # the healthy versions). Revisit when the Aqua/1.12 interaction
        # is understood.
        persistent_tasks = (broken = VERSION.major == 1 && VERSION.minor == 12,),
    )
end
