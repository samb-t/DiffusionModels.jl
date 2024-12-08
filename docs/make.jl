using Documenter, DocumenterVitepress, Pkg
using DocumenterCitations
using DiffusionModels

#! format: off

pages = [
    "Home" => "index.md",
    "Getting Started" => [
        "Introduction" => "introduction/index.md",
        # "Overview" => "introduction/overview.md",
        # "Resources" => "introduction/resources.md",
        # "Citation" => "introduction/citation.md"
    ],
    "Examples" => [
        "Overview" => "examples/index.md",
        "Lux" => [
            "examples/lux_examples/1_lux_gaussian_diffusion.md",
        ],
    ],
    "Manual" => [
        "Overview" => "manual/index.md",
    ],
    "API" => [
        "api/interface.md",
        "Gaussian Diffusion" => [
            "api/gaussian_diffusion/noise_schedules.md",
            "api/gaussian_diffusion/score_parameterisations.md",
            "api/gaussian_diffusion/gaussian_diffusion.md",
        ],
        "api/jump_diffusion.md",
        "api/categorical_diffusion.md",
    ]
]

#! format: on

# deploy_config = Documenter.auto_detect_deploy_system()
# deploy_decision = Documenter.deploy_folder(deploy_config; repo="github.com/samb-t/DiffusionModels.jl",
#     devbranch="main", devurl="dev", push_preview=true)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)
makedocs(;
    sitename="DiffusionModels",
    authors="Sam Bond-Taylor et al.",
    clean=false, # true
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[
        DiffusionModels
    ],
    linkcheck=true,
    repo="https://github.com/samb-t/DiffusionModels.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/samb-t/DiffusionModels.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="TODO",
        md_output_path=".",
        build_vitepress=false,
        # deploy_decision,
    ),
    draft=false,
    plugins=[bib],
    pages,
)

deploydocs(;
    repo="github.com/samb-t/DiffusionModels.jl.git",
    push_preview=true, target="build", devbranch="main"
)
