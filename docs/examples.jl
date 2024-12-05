#! format: off
const LUX_EXAMPLES = [
    "Lux/main.jl" => "CPU",
]

const EXAMPLES = [
    collect(enumerate(Iterators.product(["lux"], first.(LUX_EXAMPLES))))...,
]
const BACKEND_LIST = lowercase.([
    last.(LUX_EXAMPLES)...,
])
#! format: on

const BACKEND_GROUP = lowercase(get(ENV, "EXAMPLES_BACKEND_GROUP", "all"))

const BUILDKITE_PARALLEL_JOB_COUNT = parse(
    Int, get(ENV, "BUILDKITE_PARALLEL_JOB_COUNT", "-1"))

const EXAMPLES_WITH_BACKEND = if BACKEND_GROUP == "all"
    EXAMPLES
else
    EXAMPLES[BACKEND_LIST .== BACKEND_GROUP]
end

const EXAMPLES_BUILDING = if BUILDKITE_PARALLEL_JOB_COUNT > 0
    id = parse(Int, ENV["BUILDKITE_PARALLEL_JOB"]) + 1 # Index starts from 0
    splits = collect(Iterators.partition(EXAMPLES_WITH_BACKEND,
        cld(length(EXAMPLES_WITH_BACKEND), BUILDKITE_PARALLEL_JOB_COUNT)))
    id > length(splits) ? [] : splits[id]
else
    EXAMPLES_WITH_BACKEND
end

const NTASKS = min(
    parse(Int, get(ENV, "DIFFUSION_MODELS_DOCUMENTATION_NTASKS", "1")), length(EXAMPLES_BUILDING))

@info "Building EXAMPLES:" EXAMPLES_BUILDING

@info "Starting DiffusionModels Examples Build with $(NTASKS) tasks."

run(`$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=@literate -e 'import Pkg; Pkg.add(["Literate", "InteractiveUtils"])'`)

asyncmap(EXAMPLES_BUILDING; ntasks=NTASKS) do (i, (d, p))
    @info "Running Example $(i): $(p) on task $(current_task())"
    path = joinpath(@__DIR__, "..", "examples", p)
    name = "$(i)_$(first(rsplit(p, "/")))"
    output_directory = joinpath(@__DIR__, "src", "EXAMPLES", d)
    tutorial_proj = dirname(path)
    file = joinpath(dirname(@__FILE__), "run_single_tutorial.jl")

    withenv("JULIA_NUM_THREADS" => "$(Threads.nthreads())",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 ÷ NTASKS)%",
        "JULIA_PKG_PRECOMPILE_AUTO" => "0", "JULIA_DEBUG" => "Literate") do
        run(`$(Base.julia_cmd()) --startup=no --code-coverage=user --threads=$(Threads.nthreads()) --project=$(tutorial_proj) "$(file)" "$(name)" "$(output_directory)" "$(path)"`)
    end

    return
end
