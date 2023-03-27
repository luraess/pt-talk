#src # This is needed to make this run as normal Julia file
using Markdown #src

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# The accelerated pseudo-transient method

### _Solving elliptic PDEs on GPUs_

#### Ludovic Räss, _Ivan Utkin_

![ethz](./figures/ethz.png)
"""

#src ######################################################################### 
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### What we will cover in this lecture:
- Why to solve PDEs on GPUs
- Brief recap on PDEs
- Solving Parabolic and hyperbolic PDEs
- The challenge: solving elliptic PDEs on GPUs
- The accelerated pseudo-transient method
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
### We will restrict us to:
- Finite-difference discretisation (stencil computations)
- Linear PDEs
- 1D problems
"""

#src ######################################################################### 
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
A **partial differential equation (PDE)** is an equation which imposes relations between the various partial derivatives of a multivariable function. [_Wikipedia_](https://en.wikipedia.org/wiki/Partial_differential_equation)
"""

#src One slide on PDEs on GPUS


#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
> _**Classification of second-order PDEs:**_
>  - **Parabolic:**\
>    $∂u/∂t - α ∇^2 u - b = 0$ (e.g. transient heat diffusion)
>  - **Hyperbolic:**\
>    $∂^2u/∂t^2 - c^2 ∇^2 u = 0$ (e.g. acoustic wave equation)
>  - **Elliptic:**\
>    $∇^2 u - b = 0$ (e.g. steady state diffusion, Laplacian)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Parabolic PDEs - diffusion
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
### The diffusion equation

<center>
  <video width="80%" autoplay loop controls src="./figures/diffusion_1D.mp4"/>
</center>
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The diffusion equation is often reported as a second order parabolic PDE, here for a multivariable function $C(x,t)$ showing derivatives in both temporal $∂t$ and spatial $∂x$ dimensions (here for the 1D case)

$$
\frac{∂C}{∂t} = D\frac{∂^2 C}{∂ x^2}~,
$$

where $D$ is the diffusion coefficient.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
A more general description combines a diffusive flux:

$$ q = -D\frac{∂C}{∂x}~,$$
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
and a conservation or flux balance equation:

$$ \frac{∂C}{∂t} = -\frac{∂q}{∂x}~. $$
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
A concise and efficient solution to the diffusion equation on GPUs can be achieved combining an explicit time integration and using a finite-difference spatial discretisation (stencil computations).
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Let's recall that solving explicit wave propagation on GPUs is where it all started, mostly ([Micikevicius 2009](https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf)).
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Hyperbolic PDEs - acoustic wave propagation
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
### The wave equation

<center>
  <video width="80%" autoplay loop controls src="./figures/acoustic_1D.mp4"/>
</center>
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The [wave equation](https://en.wikipedia.org/wiki/Wave_equation) is a second-order linear partial differential equation for the description of waves —as they occur in classical physics— such as mechanical waves (e.g. water waves, sound waves and seismic waves) or light waves.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The hyperbolic equation reads

$$ \frac{∂^2P}{∂t^2} = c^2 ∇^2 P~,$$

where
- $P$ is pressure (or, displacement, or another scalar quantity...)
- $c$ a real constant (speed of sound, stiffness, ...)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### From diffusion to acoustic wave propagation

The hyperbolic (wave) equation can also be written as a first order system, similar to the one that we used to implement the diffusion equation.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
$$ \frac{∂^2 P}{∂t^2} = c^2 ∇^2 P~,$$

as two first order equations 

$$ \frac{∂V_x}{∂t} = -\frac{1}{ρ}~\frac{∂P}{∂x}~,$$

$$ \frac{∂P}{∂t}  = -\frac{1}{\beta}~\frac{∂V_x}{∂x}~.$$
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Comparing the equations

Comparing diffusive and wave physics, we can summarise following:

| Diffusion                                                            | Wave propagation                                                                    |
|:--------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|
| $$ q = -D\frac{\partial C}{\partial x} $$                            | $$ \frac{\partial V_x}{\partial t} = -\frac{1}{\rho}\frac{\partial P}{\partial x} $$  |
| $$ \frac{\partial C}{\partial t} = -\frac{\partial q}{\partial x} $$ | $$ \frac{\partial P}{\partial t} = -\frac{1}{\beta}\frac{\partial V_x}{\partial x} $$ |

👉 We see that the main difference is the update instead of the assignment of the "flux" in the wave propagation
"""



#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Towards solving the elliptic problem

We have considered numerical solutions to the hyperbolic and parabolic PDEs.

👉 In both cases we used the explicit time integration
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The elliptic PDE is different:

$$
\frac{\partial^2 C}{\partial x^2} = 0
$$

It doesn't depend on time! How do we solve it numerically then?
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### A first solution to the elliptic PDE
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The solution of an elliptic PDE is actually the steady state limit of the time-dependent diffusion problem described by the parabolic PDE:

$$
\frac{\partial^2 C}{\partial x^2} - \frac{\partial C}{\partial t} = 0
$$

when $t\rightarrow\infty$, and we know how to solve parabolic PDEs.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Increasing the number of time steps `nt` in our diffusion code will allow the solution to converge towards a steady state:

<center>
  <video width="80%" autoplay loop controls src="./figures/diffusion_1D_steady_state.mp4"/>
</center>
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
We approach the steady-state, but the number of time steps required to converge to a solution is proportional to `nx^2`.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
- For simulations in 1D and low resolutions in 2D the quadratic scaling is acceptable.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
- For high-resolution 2D and 3D the `nx^2` factor becomes prohibitively expensive!
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
But we know how to handle this 🚀
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Solving elliptic PDEs

We just established that the solution to the elliptic PDE could be obtained through integrating in time a corresponding parabolic PDE:

$$
\frac{\partial C}{\partial t} - \frac{\partial^2 C}{\partial x^2} = 0
$$
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
with the major limitations of this approach being the quadratic dependence of the number of time steps on the number of grid points in spatial discretisation.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Accelerating elliptic solver convergence: intuition

We'll now improve the convergence rate of the elliptic solver (which can be generalised to higher dimensions).
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Let's recall the stability conditions for diffusion and acoustic wave propagation:

```julia
dt = dx^2/dc/2      # diffusion
dt = dx/sqrt(1/β/ρ) # acoustic wave propagation
```
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
We can see that the acceptable time step for an acoustic problem is proportional to the grid spacing `dx`, and not `dx^2` as for the diffusion.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The number of time steps required for the wave to propagate through the domain is only proportional to the number of grid points `nx`.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Can we use that information to reduce the time required for the elliptic solver to converge?
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
In the solution to the wave equation, the waves do not attenuate with time: _there is no steady state!_

<center>
  <video width="80%" autoplay loop controls src="./figures/acoustic_1D.mp4"/>
</center>
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Damped wave equation

Let's add diffusive properties to the wave equation by simply combining the physics:

\begin{align}
\rho\frac{\partial V_x}{\partial t}                 &= -\frac{\partial P}{\partial x} \nonumber \\[10pt]
\beta\frac{\partial P}{\partial t} + \frac{P}{\eta} &= -\frac{\partial V_x}{\partial x} \nonumber
\end{align}
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Note the addition of the new term $\frac{P}{\eta}$ to the left-hand side of the mass balance equation, which could be interpreted physically as accounting for the bulk viscosity of the gas.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Equivalently, we could add the time derivative to the diffusion equation

\begin{align}
\rho\frac{\partial q}{\partial t} + \frac{q}{D} &= -\frac{\partial C}{\partial x} \nonumber \\[10pt]
\frac{\partial C}{\partial t}                   &= -\frac{\partial q}{\partial x} \nonumber
\end{align}
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
In that case, the new term would be $\rho\frac{\partial q}{\partial t}$, which could be interpreted physically as adding the inertia to the momentum equation for diffusive flux.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
> 💡 Note: In 1D, both modifications are equivalent up to renaming the variables. The conceptual difference is that in the former case we add new terms to the vector quantity (diffusive flux $q$), and in the latter case we modify the equation governing the evolution of the scalar quantity (pressure $P$).
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Let's eliminate $V_x$ and $q$ in both systems to get one governing equation for $P$ and $C$, respectively:
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
\begin{align}
\beta\frac{\partial^2 P}{\partial t^2} + \frac{1}{\eta}\frac{\partial P}{\partial t} &= \frac{1}{\rho}\frac{\partial^2 P}{\partial x^2} \nonumber \\[10pt]
\rho\frac{\partial^2 C}{\partial t^2} + \frac{1}{D}\frac{\partial C}{\partial t}     &= \frac{\partial^2 C}{\partial x^2} \nonumber
\end{align}
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
We refer to such equations as the _**damped wave equations**_. They combine wave propagation with diffusion, which manifests as wave attenuation, or decay. The damped wave equation is a hyperbolic PDE.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
<center>
  <video width="80%" autoplay loop controls src="./figures/damped_diffusion_1D.mp4"/>
</center>
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The waves decay, now there is a steady state! 🎉 The time it takes to converge, however, doesn't seem to improve...
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Solve the hyperbolic PDE with the implicit flux term treatment, the time step should become proportional to the grid spacing `dx` instead of `dx^2`.

Looking at the damped wave equation for $C$, and recalling the stability condition for wave propagation, we can modify the time step to the following:

```julia
dt   = dx/sqrt(1/ρ)
```
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Re-running the simulation we now get:

<center>
  <video width="80%" autoplay loop controls src="./figures/damped_diffusion_better_1D.mp4"/>
</center>
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Now, this is much better! We observe that in less time steps, we get a much faster convergence. However, we introduced the new parameter, $\rho$. How does the solution depend on the value of $\rho$?
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Problem of finding the iteration parameters

Changing the new parameter `ρ`, what happens to the solution?
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
![vary rho](./figures/vary_rho_2.png)

We notice that depending on the value of the parameter `ρ`, the convergence to steady-state can be faster or slower. If `ρ` is too small, the process becomes diffusion-dominated, if `ρ` is too large, waves decay slowly.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
If the parameter `ρ` has optimal value, the convergence to steady-state could be achieved in the number of time steps proportional to the number of grid points `nx` and not `nx^2` as for the parabolic PDE.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
For linear PDEs it is possible to determine the optimal value for `ρ` analytically:
```julia
ρ = (lx/(dc*2π))^2
```
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
How does one derive the optimal values for other problems and boundary conditions?
Unfortunately, we don't have time to dive into details now...

The idea of accelerating the convergence by increasing the order of PDE dates back to the work by [Frankel (1950)](https://doi.org/10.2307/2002770) where he studied the convergence rates of different iterative methods. Frankel noted the analogy between the iteration process and transient physics. In his work, the accelerated method was called the _second-order Richardson method_.

👀 If interested, [Räss et al. (2022)](https://gmd.copernicus.org/articles/15/5757/2022/) paper is a good starting point.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Pseudo-transient method

We can thus call any method that builds upon the analogy to the transient physics the _pseudo-transient_ method.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Using this analogy proves useful when studying multi-physics and nonlinear processes. The pseudo-transient method isn't restricted to solving the Poisson problems, but can be applied to a wide range of problems that are modelled with PDEs.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
In a pseudo-transient method, we are interested only in a steady-state distributions of the unknown field variables such as concentration, temperature, etc.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
We consider time steps as iterations in a numerical method. Therefore, we replace the time $t$ in the equations with _pseudo-time_ $\tau$, and a time step `it` with iteration counter `iter`. When a pseudo-transient method converges, all the pseudo-time derivatives $\partial/\partial\tau$, $\partial^2/\partial\tau^2$ etc., vanish.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
> ⚠️ Warning: We should be careful when introducing the new pseudo-physical terms into the governing equations. We need to make sure that when iterations converge, i.e., if the pseudo-time derivatives are set to 0, the system of equations is identical to the original steady-state formulation.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
For example, consider the damped acoustic problem that we introduced in the beginning:

\begin{align}
\rho\frac{\partial V_x}{\partial\tau}                 &= -\frac{\partial P}{\partial x} \nonumber \\[10pt]
\beta\frac{\partial P}{\partial\tau} + \frac{P}{\eta} &= -\frac{\partial V_x}{\partial x} \nonumber
\end{align}
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
At the steady-state, the second equation reads:

$$
\frac{P}{\eta} = -\frac{\partial V_x}{\partial x}
$$

The velocity divergence is proportional to the pressure. If we wanted to solve the incompressible problem (i.e. the velocity divergence = 0), and were interested in the velocity distribution, this approach would lead to incorrect results. Only add new terms to the governing equations that vanish when the iterations converge!
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Visualising convergence

The final addition to the simple elliptic solver is to monitor convergence and stop iterations when the error has reached predefined tolerance.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
To define the measure of error, we introduce the residual:

$$
r_C = D\frac{\partial^2 \widehat{C}}{\partial x^2}
$$

where $\widehat{C}$ is the pseudo-transient solution.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
There are many ways to define the error as the norm of the residual, the most popular ones are the $L_2$ norm and $L_\infty$ norm. We can here use the $L_\infty$ norm:

$$
\|\boldsymbol{r}\|_\infty = \max_i(|r_i|)
$$
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
<center>
  <video width="80%" autoplay loop controls src="./figures/converge_diffusion_1D.mp4"/>
</center>
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Wrapping-up

- Switching from parabolic to hyperbolic PDE allows to approach the steady-state in number of iterations, proportional to the number of grid points
- Pseudo-transient (PT) method is the matrix-free iterative method to solve elliptic (and other) PDEs by utilising the analogy to transient physics
- Using the optimal iteration parameters is essential to ensure the fast convergence of the PT method
- Extending the codes to 2D and 3D is straightforward with explicit time integration
"""
