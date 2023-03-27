using Plots, Plots.Measures, Printf
default(size=(1200, 400), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function damped_diffusion_1D()
    # physics
    lx   = 20.0
    dc   = 1.0
    ρ    = 1000
    # ρ    = (lx/(dc*2π))^2
    # numerics
    nx   = 200
    nvis = 10
    # derived numerics
    dx   = lx / nx
    dt   = dx / sqrt(1 / ρ)
    nt   = 5nx
    # dt   = dx^2/dc/2
    # nt   = nx^2 ÷ 5
    xc   = LinRange(dx / 2, lx - dx / 2, nx)
    # array initialisation
    C    = @. 1.0 + exp(-(xc - lx / 4)^2) - xc / lx; C_i = copy(C)
    qx   = zeros(Float64, nx - 1)
    # time loop
    # ispath("anim") && rm("anim", recursive=true); mkdir("anim"); iframe = -1
    for it = 1:nt
        qx         .-= dt ./ (ρ * dc .+ dt) .* (qx .+ dc .* diff(C) ./ dx)
        C[2:end-1] .-= dt .* diff(qx) ./ dx
        if it % nvis == 0
            sleep(0.02); display(plot(xc, [C_i, C]; xlims=(0, lx), ylims=(-0.1, 2.0),
                xlabel="lx", ylabel="Concentration", title="time = $(round(it*dt,digits=1))"))
        end
    end
end

damped_diffusion_1D()