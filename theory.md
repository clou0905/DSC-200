## 1. Aerosols: Sources and Physical Processes

Atmospheric aerosols are small solid or liquid particles suspended in the atmosphere.
PM2.5 refers to particulate matter with an aerodynamic diameter of 2.5 μm or less. Its concentration and distribution are influenced by natural and anthropogenic emissions, atmospheric transport, chemical transformations, and deposition.
This model integrates machine learning (ML) with a physics-based advection–diffusion–reaction equation to simulate the spatiotemporal evolution of global PM2.5 concentration.

## 2. Governing Equation

### 2.1 Advection–Diffusion–Reaction Equation

Let $\Omega \subset \mathbb{R}^2$ be a spatial domain in two-dimensional Euclidean space.

$$
\frac{\partial C}{\partial t}
+\mathbf{u}\cdot\nabla C
=\nabla\cdot(K\nabla C)+L-\lambda C.
$$

where:

| Parameter | Description |
| --- | --- |
| $C(x,t)$ | PM2.5 mass concentration |
| $\mathbf{x}=(x,y)$ | Spatial coordinates |
| $t$ | Time |
| $\mathbf{u}=(u,v)$ | Horizontal wind velocity |
| $K$ | Horizontal diffusion coefficient |
| $L$ | Source term |
| $\lambda$ | First-order removal coefficient |

### 2.2 Advection

Advection describes the transport of PM2.5 by atmospheric wind.

$$
\mathbf{u}\cdot\nabla C
$$

The wind velocity determines the direction and rate of transport.

### 2.3 Diffusion

Diffusion describes the spatial redistribution of PM2.5 due to atmospheric turbulence.

$$
\nabla\cdot(K\nabla C)
$$

The diffusion coefficient $K$ determines the strength of mixing and may vary in both space and time.

### 2.4 Source and Sink

The source and sink terms describe the local production and removal of PM2.5.

$$
L-\lambda C
$$

The source term $L$ represents the production of PM2.5, while $\lambda$ determines the first-order removal rate.


### 2.3 Machine Learning Parameterization

The diffusion coefficient K, source term L, and removal coefficient λ are influenced by various atmospheric processes and meteorological conditions.



#### Diffusion Coefficient (K)

The diffusion coefficient represents the strength of atmospheric turbulent mixing. It may depend on wind speed, atmospheric stability, boundary layer height, and surface heating.



#### Source Term (L)

The source term represents primary emissions, secondary aerosol formation, and aerosol transport into the surface layer. It may depend on temperature, humidity, solar radiation, boundary layer height, and emission intensity.



#### Removal Coefficient (λ)

The removal coefficient represents the loss of PM2.5 through dry deposition, wet deposition, and transport out of the surface layer. It may depend on precipitation, humidity, wind speed, and boundary layer height.

These complex relationships can be approximated using machine learning models.

These processes are influenced by the following atmospheric variables:

| Variable | Physical Relevance |
| --- | --- |
| PM2.5 | Concentration-dependent removal and transformation |
| u10, v10 | Wind transport and turbulent mixing |
| t2m | Temperature-dependent chemical reactions |
| d2m | Atmospheric moisture and aerosol formation |
| blh | Vertical mixing and dilution |
| tp | Wet deposition through precipitation |
| ssrd | Photochemical reactions and surface heating |

Machine learning is used to approximate the complex relationships between these atmospheric variables and the unresolved physical processes, rather than directly replacing the governing equation.

## 3. Spherical Geometry

Atmospheric transport is modeled on a two-dimensional spherical surface to account for the curvature of the Earth.

### 3.1 Spherical Coordinates

Let $R = 6.371 \times 10^6$ m be the Earth's radius.
A point on the sphere is parameterized by longitude
$\theta$ and latitude $\phi$:

$$
\mathbf{x}(\theta,\phi)
=
R
\begin{pmatrix}
\cos\phi\cos\theta \\
\cos\phi\sin\theta \\
\sin\phi
\end{pmatrix},
$$

where

$$
\theta \in [0,2\pi), \qquad
\phi \in \left[-\frac{\pi}{2},\frac{\pi}{2}\right].
$$

The spherical surface area element is

$$
dA = R^2\cos\phi\,d\theta\,d\phi,
$$

which will be used in the surface integrals of the weak formulation.

### 3.2 Spherical Differential Operators

Let $\mathbf{e}_\theta$ and $\mathbf{e}_\phi$ denote the eastward and northward unit tangent vectors, respectively.

**Surface Gradient**

$$
\nabla_S C
=
\frac{1}{R\cos\phi}
\frac{\partial C}{\partial\theta}\mathbf{e}_\theta
+
\frac{1}{R}
\frac{\partial C}{\partial\phi}\mathbf{e}_\phi.
$$

**Surface Divergence**

For a tangential vector field

$$
\mathbf{F}
=
F_\theta\mathbf{e}_\theta
+
F_\phi\mathbf{e}_\phi,
$$

the surface divergence is

$$
\nabla_S\cdot\mathbf{F}
=
\frac{1}{R\cos\phi}
\left[
\frac{\partial F_\theta}{\partial\theta}
+
\frac{\partial(F_\phi\cos\phi)}{\partial\phi}
\right].
$$

### 3.3 Governing Equation on the Sphere

The spherical advection–diffusion–reaction equation is

$$
\frac{\partial C}{\partial t}
+
\mathbf{u}\cdot\nabla_S C
=
\nabla_S\cdot(K\nabla_S C)
+
S-\lambda C,
$$

where

$$
\mathbf{u}
=
u_\theta\mathbf{e}_\theta
+
u_\phi\mathbf{e}_\phi.
$$

**Spherical Advection**

$$
\mathbf{u}\cdot\nabla_S C
=
\frac{u_\theta}{R\cos\phi}
\frac{\partial C}{\partial\theta}
+
\frac{u_\phi}{R}
\frac{\partial C}{\partial\phi}.
$$

**Spherical Diffusion**

Substituting $\mathbf{F}=K\nabla_S C$ into the surface divergence gives

$$
\nabla_S\cdot(K\nabla_S C)
=
\frac{1}{R^2\cos^2\phi}
\frac{\partial}{\partial\theta}
\left(K\frac{\partial C}{\partial\theta}\right)
+
\frac{1}{R^2\cos\phi}
\frac{\partial}{\partial\phi}
\left(K\cos\phi\frac{\partial C}{\partial\phi}\right).
$$

The spherical metric factors are incorporated into the weak formulation and finite element discretization.

### 3.4 Notation Reference

| Symbol | Description |
| --- | --- |
| $R$ | Earth's radius |
| $\theta, \phi$ | Longitude and latitude (radians) |
| $\mathbf{e}_\theta, \mathbf{e}_\phi$ | Eastward and northward unit tangent vectors |
| $\mathbf{u}$ | Horizontal wind velocity |
| $u_\theta, u_\phi$ | Eastward and northward wind components |
| $\mathbf{F}$ | Tangential vector field |
| $F_\theta, F_\phi$ | Eastward and northward components of $\mathbf{F}$ |
| $\nabla_S$ | Surface gradient operator |
| $\nabla_S\cdot$ | Surface divergence operator |
| $dA$ | Spherical surface area element |

## 4. Weak Formulation

### 4.1 Derivation of the Weak Form

Consider the advection–diffusion–reaction equation on a two-dimensional Euclidean domain $\Omega\subset\mathbb{R}^2$:

$$
\frac{\partial C}{\partial t}
+\mathbf{u}\cdot\nabla C
=
\nabla\cdot(K\nabla C)+S-\lambda C.
$$

Multiplying the equation by a test function $v$, integrating over $\Omega$, and applying integration by parts to the diffusion term gives

$$
\int_\Omega v\nabla\cdot(K\nabla C)\,d\mathbf{x}
=
-\int_\Omega K\nabla v\cdot\nabla C\,d\mathbf{x}
+\int_{\partial\Omega}vK\nabla C\cdot\mathbf{n}\,ds.
$$

Assuming a zero diffusive flux boundary condition,

$$
K\nabla C\cdot\mathbf{n}=0
\qquad\text{on }\partial\Omega,
$$

we obtain the weak formulation:

$$
\int_\Omega v\frac{\partial C}{\partial t}\,d\mathbf{x}
+\int_\Omega v(\mathbf{u}\cdot\nabla C)\,d\mathbf{x}
+\int_\Omega K\nabla v\cdot\nabla C\,d\mathbf{x}
=
\int_\Omega v(S-\lambda C)\,d\mathbf{x}.
$$

The weak formulation reduces the spatial differentiability requirement of $C$ by transferring one derivative from $C$ to the test function $v$.

### 4.2 Choice of Test Functions

Let $V=H^1(\Omega)$ be the test function space.
The weak formulation must hold for every test function $v\in V$.

For numerical approximation, we introduce a finite-dimensional subspace $V_h\subset V$:

$$
V_h=\operatorname{span}\{N_1,N_2,\ldots,N_M\},
$$

where $N_i$ are the basis functions and $M$ is the number of mesh nodes.

The approximate solution is expressed as

$$
C_h(\mathbf{x},t)
=
\sum_{j=1}^{M}C_j(t)N_j(\mathbf{x}).
$$

Following the Galerkin method, the test functions are chosen from the same basis functions:

$$
v=N_i,\qquad i=1,2,\ldots,M.
$$

This converts the continuous weak formulation into a finite-dimensional system of equations.

In this model, continuous piecewise linear (P1) basis functions are used as test functions. The specific construction and selection of these test functions will be explained in detail in the spatial discretization section.

### 4.3 Comparison of Strong and Weak Formulations

| Aspect | Strong Formulation | Weak Formulation |
| --- | --- | --- |
| Computational cost | Numerical approximation of second-order derivatives | First-order derivatives and numerical integration, with additional assembly costs |
| Numerical approximation | Commonly discretized using finite differences | Supports continuous piecewise linear basis functions, but requires mesh construction and integration |
| Parallel computing | Supports grid-based node-wise parallel computation | Supports independent element-wise computation, followed by global assembly |
| Solution regularity | Requires higher spatial regularity | Accommodates solutions with limited spatial regularity |
| Numerical stability | Depends on the discretization and time-stepping scheme | Depends on the discretization and time-stepping scheme; may require additional stabilization |

The following table compares the strong formulation using finite differences with the weak formulation using finite element methods.

## 5. Spatial Discretization

To numerically solve the weak formulation, the spherical domain is discretized into a triangular mesh using the finite element method (FEM).

### 5.1 Spatial Grid

The Earth's surface is discretized using a regular latitude–longitude grid with a spatial resolution of

$$
\Delta\theta=\Delta\phi=0.75^\circ.
$$

The grid consists of 480 longitude points and 241 latitude points, defined by

$$
\theta_i=i\Delta\theta,
\qquad i=0,1,\ldots,479,
$$

$$
\phi_j=\frac{\pi}{2}-j\Delta\phi,
\qquad j=0,1,\ldots,240.
$$

Here, angular quantities are expressed in radians when used in numerical calculations.

Although the grid is uniform in angular coordinates, the physical distance between neighboring longitude points varies with latitude:

$$
h_\theta=R\cos\phi\,\Delta\theta,
\qquad
h_\phi=R\Delta\phi.
$$

Therefore, spherical geometry must be taken into account when evaluating spatial derivatives and surface integrals.

### 5.2 Triangular Mesh

Each rectangular cell in the latitude–longitude grid is divided into two triangular elements.

Let the four vertices of a rectangular cell be

$$
\begin{aligned}
A&=(\theta_i,\phi_j),\\
B&=(\theta_{i+1},\phi_j),\\
C&=(\theta_i,\phi_{j+1}),\\
D&=(\theta_{i+1},\phi_{j+1}).
\end{aligned}
$$

The cell is divided into two triangular elements:

$$
T_1=\triangle ABC,
\qquad
T_2=\triangle BCD.
$$

The computational domain is represented as a collection of non-overlapping triangular elements:

$$
\overline{\Omega}_h
=
\bigcup_{e=1}^{N_e}\overline{T}_e,
$$

where $N_e$ is the total number of triangular elements.

The integral of a function over the computational domain can therefore be decomposed into element-level integrals:

$$
\int_{\Omega_h} f\,dA
=
\sum_{e=1}^{N_e}
\int_{T_e} f\,dA.
$$

This allows element-level contributions to be calculated independently before being assembled into the global system.

### 5.3 Triangular Gaussian Quadrature

#### 5.3.1 General Formulation

Gaussian quadrature approximates an integral over a
triangular domain $T$ using a weighted sum of function
values at selected quadrature points:

$$
\int_T f(\mathbf{x})\,d\mathbf{x}
\approx
|T|\sum_{q=1}^{Q}w_qf(\mathbf{x}_q),
$$

where:

| Parameter | Description |
| --- | --- |
| $\lvert T\rvert$ | Area of the triangular element |
| $Q$ | Number of quadrature points |
| $\mathbf{x}_q$ | The $q$-th quadrature point |
| $w_q$ | Normalized weight of the $q$-th quadrature point |

The normalized weights satisfy

$$
\sum_{q=1}^{Q}w_q=1.
$$

In this model, a three-point quadrature rule is used,
with quadrature points located at the midpoints of
the three edges:

$$
\begin{aligned}
\mathbf{x}_{AB}&=\frac{A+B}{2},\\
\mathbf{x}_{BC}&=\frac{B+C}{2},\\
\mathbf{x}_{CA}&=\frac{C+A}{2}.
\end{aligned}
$$

The corresponding weights are

$$
w_{AB}=w_{BC}=w_{CA}=\frac{1}{3}.
$$

Therefore,

$$
\int_T f(\mathbf{x})\,d\mathbf{x}
\approx
\frac{|T|}{3}
\left[
f(\mathbf{x}_{AB})
+f(\mathbf{x}_{BC})
+f(\mathbf{x}_{CA})
\right].
$$

This quadrature rule is exact for polynomials of
total degree at most two.

#### 5.3.2 Quadrature on the Spherical Surface

Let $T$ be a triangular element on the spherical
surface and $\widehat{T}$ its corresponding triangle
in the longitude–latitude parameter space.

Using the spherical surface area element

$$
dA=R^2\cos\phi\,d\theta\,d\phi,
$$

the surface integral can be expressed as

$$
\int_T f\,dA
=
\int_{\widehat{T}}
f(\theta,\phi)R^2\cos\phi\,d\theta\,d\phi.
$$

Applying the triangular Gaussian quadrature rule
in the parameter space gives

$$
\int_T f\,dA
\approx
R^2|\widehat{T}|
\sum_{q=1}^{Q}
w_q f(\theta_q,\phi_q)\cos\phi_q.
$$

For the three-point quadrature rule,

$$
\int_T f\,dA
\approx
\frac{R^2|\widehat{T}|}{3}
\left[
f_{AB}\cos\phi_{AB}
+f_{BC}\cos\phi_{BC}
+f_{CA}\cos\phi_{CA}
\right],
$$

where $f_{AB}$, $f_{BC}$, and $f_{CA}$ denote the
function values at the corresponding edge midpoints.

#### 5.3.3 Application to the Triangular Mesh

Consider the triangular element with vertices

$$
\begin{aligned}
A&=(\theta_i,\phi_j),\\
B&=(\theta_i+\Delta\theta,\phi_j),\\
C&=(\theta_i,\phi_j-\Delta\phi).
\end{aligned}
$$

The area of the triangle in the parameter space is

$$
|\widehat{T}|=\frac{\Delta\theta\Delta\phi}{2}.
$$

The three quadrature points are

$$
\begin{aligned}
\mathbf{x}_{AB}
&=
\left(
\theta_i+\frac{\Delta\theta}{2},
\phi_j
\right),\\
\mathbf{x}_{BC}
&=
\left(
\theta_i+\frac{\Delta\theta}{2},
\phi_j-\frac{\Delta\phi}{2}
\right),\\
\mathbf{x}_{CA}
&=
\left(
\theta_i,
\phi_j-\frac{\Delta\phi}{2}
\right).
\end{aligned}
$$

Substituting these quadrature points and the
parameter-space area into the spherical quadrature
formula yields

$$
\int_T f\,dA
\approx
\frac{R^2\Delta\theta\Delta\phi}{6}
\left[
f_{AB}\cos\phi_j
+(f_{BC}+f_{CA})\cos\left(\phi_j-\frac{\Delta\phi}{2}\right)
\right].
$$

The same quadrature procedure is applied to the
second triangular element of each grid cell.

### 5.4 Element-Level Weak Formulation

Using the conforming triangular mesh introduced above,
the weak formulation can be decomposed into a sum of
element-level integrals:

$$
\sum_{e=1}^{N_e}
\left[
\int_{T_e}v\frac{\partial C}{\partial t}\,dA
+\int_{T_e}v(\mathbf{u}\cdot\nabla_S C)\,dA
+\int_{T_e}K\nabla_S v\cdot\nabla_S C\,dA
-\int_{T_e}v(L-\lambda C)\,dA
\right]
=0.
$$

Each element-level integral is evaluated using local basis functions and the triangular quadrature rule introduced in Section 5.3.

The following sections describe the discretization of each term over a single triangular element.