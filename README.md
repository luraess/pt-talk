# Pseudo-transient method - talk
Birds view of the accelerated pseudo-transient method

## Automatic notebook generation

The presentation slides and the demo notebook base on the [Julia programming language](https://julialang.org). The material is self-contained in a Jupyter notebook [pt_lecture.ipynb](pt_lecture.ipynb) that can be auto-generated using literate programming by deploying the [pt_lecture.jl](pt_lecture.jl) script.

To reproduce:
1. Clone this git repo
2. Open Julia and resolve/instantiate the project
```julia-repl
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
```
3. Run the deploy script
```julia-repl
julia> using Literate

julia> include("deploy_notebooks.jl")
```
4. Then using IJulia, you can launch the notebook and get it displayed in your web browser:
```julia-repl
julia> using IJulia

julia> notebook(dir="./")
```
_To view the notebook as slide, you need to install the [RISE](https://rise.readthedocs.io/en/stable/installation.html) plugin_

## Resources
#### Courses and resources
- GMD 2022 paper on the accelerated pseudo-transient method: [DOI: 10.5194/gmd-15-5757-2022](https://doi.org/10.5194/gmd-15-5757-2022)
- ETHZ course on solving PDEs with GPUs: https://pde-on-gpu.vaw.ethz.ch
- More [here](https://pde-on-gpu.vaw.ethz.ch/extras/#extra_material)

#### Misc
- Frontier GPU multi-physics solvers: https://ptsolvers.github.io/GPU4GEO/
