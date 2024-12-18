@testsnippet SharedTestSetup begin

    using Test
    using JET
    using RequiredInterfaces: check_interface_implemented
    using Random
    using Distributions
    using Statistics
    using SciMLBase
    using StochasticDiffEq
    using TestItemRunner

    using DiffusionModels

    using JET: JET, JETTestFailure, get_reports, report_call, report_opt
    # XXX: In 1.11, JET leads to stack overflows
    # global JET_TESTING_ENABLED = v"1.10-" ≤ VERSION < v"1.11-"
    global JET_TESTING_ENABLED = true # hope for the best...

end
