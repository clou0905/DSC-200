## 1. Aerosols: Sources and Physical Processes

Atmospheric aerosols are small solid or liquid particles suspended in the atmosphere. PM2.5 refers to particulate matter with an aerodynamic diameter of 2.5 μm or less. Its concentration and distribution are influenced by natural and anthropogenic emissions, atmospheric transport, chemical transformations, and deposition. This model integrates machine learning (ML) with a physics-based advection–diffusion–reaction equation to simulate the spatiotemporal evolution of global PM2.5 concentration.

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

### 2.5 Machine Learning Parameterization

The diffusivity $K$ and the coefficients $L$ and $\lambda$ are predicted at each grid cell. They do not replace the governing equation. Advection still uses the resolved wind.

Both networks read the same four fields: boundary-layer height (blh), 2 m temperature (t2m), 2 m dewpoint (d2m), and surface solar radiation (ssrd). The 10 m wind components u10 and v10 are the eastward and northward velocities in the advection term, and they are not network inputs. Total precipitation is not used. PM2.5 enters only through the linear sink $\lambda C$.

EAC4 stores PM2.5 in kg m$^{-3}$; the model converts it to $\mu$g m$^{-3}$. ERA5 stores ssrd in J m$^{-2}$ over the 3-hour window; the model divides it by $\Delta t$ to obtain W m$^{-2}$.

Each network is a multilayer perceptron with three hidden layers of width 16 and SiLU activations. The diffusivity network has one output $f_K$. The source network has two outputs $f_L$ and $f_\lambda$. Learned per-cell scales $k_s$, $L_s$, and $\lambda_s$ multiply these outputs. With $\Delta t=10800$ s,

$$
K=\mathrm{softplus}(k_s f_K),
$$

$$
L=\mathrm{softplus}(L_s f_L-4)/\Delta t,
$$

$$
\lambda=\mathrm{sigmoid}(\lambda_s f_\lambda-4)(0.5/\Delta t).
$$

Thus $K\ge 0$, $L\ge 0$, and $0\le\lambda\le 0.5/\Delta t$. The tendency passed to the time step is $L-\lambda C$, in $\mu$g m$^{-3}$ s$^{-1}$.

| Variable | Role in the model |
| --- | --- |
| blh, t2m, d2m, ssrd | Inputs of both networks |
| u10, v10 | Eastward and northward 10 m wind in the advection term |
| PM2.5 | Concentration $C$ in the sink $\lambda C$ |

## 3. Spherical Geometry

Atmospheric transport is modeled on a two-dimensional spherical surface to account for the curvature of the Earth.

### 3.1 Spherical Coordinates

Let $R = 6.371 \times 10^6$ m be the Earth's radius. A point on the sphere is parameterized by longitude $\theta$ and latitude $\phi$:

```math
\mathbf{x}(\theta,\phi)
=
R
\begin{pmatrix}
\cos\phi\cos\theta \\
\cos\phi\sin\theta \\
\sin\phi
\end{pmatrix},
```

where

$$
\theta \in [0,2\pi), \qquad
\phi \in \left[-\frac{\pi}{2},\frac{\pi}{2}\right].
$$

The spherical surface area element is

$$
dA = R^2\cos\phi d\theta d\phi,
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
L-\lambda C,
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
| $\mathbf{e}_\theta$, $\mathbf{e}_\phi$ | Eastward and northward unit tangent vectors |
| $\mathbf{u}$ | Horizontal wind velocity |
| $u_\theta$, $u_\phi$ | Eastward and northward wind components |
| $\mathbf{F}$ | Tangential vector field |
| $F_\theta$, $F_\phi$ | Eastward and northward components of $\mathbf{F}$ |
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
\nabla\cdot(K\nabla C)+L-\lambda C.
$$

Multiplying the equation by a test function $v$, integrating over $\Omega$, and applying integration by parts to the diffusion term gives

$$
\int_\Omega v\nabla\cdot(K\nabla C) d\mathbf{x}
=
-\int_\Omega K\nabla v\cdot\nabla C d\mathbf{x}
+\int_{\partial\Omega}vK\nabla C\cdot\mathbf{n} ds.
$$

Assuming a zero diffusive flux boundary condition,

$$
K\nabla C\cdot\mathbf{n}=0
\qquad\text{on }\partial\Omega,
$$

we obtain the weak formulation:

$$
\int_\Omega v\frac{\partial C}{\partial t} d\mathbf{x}
+\int_\Omega v(\mathbf{u}\cdot\nabla C) d\mathbf{x}
+\int_\Omega K\nabla v\cdot\nabla C d\mathbf{x}
=
\int_\Omega v(L-\lambda C) d\mathbf{x}.
$$

The weak formulation reduces the spatial differentiability requirement of $C$ by transferring one derivative from $C$ to the test function $v$.

The calculation above is the Euclidean derivation. The model itself is posed on a closed spherical surface, so there is no boundary integral. Longitude is periodic.

### 4.2 Choice of Test Functions

Let $V=H^1(\Omega)$ be the test function space. The weak formulation must hold for every test function $v\in V$.

For numerical approximation, we introduce a finite-dimensional subspace $V_h\subset V$:

$$
V_h=\mathrm{span}\lbrace N_1,N_2,\ldots,N_M\rbrace,
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

In this model, continuous piecewise linear (P1) basis functions are used as test functions. Their derivatives on each triangle are given in Sections 5.5 and 5.6.

### 4.3 Comparison of Strong and Weak Formulations

The following table compares the strong formulation using finite differences with the weak formulation using finite element methods.

| Aspect | Strong Formulation | Weak Formulation |
| --- | --- | --- |
| Computational cost | Numerical approximation of second-order derivatives | First-order derivatives and numerical integration, with additional assembly costs |
| Numerical approximation | Commonly discretized using finite differences | Supports continuous piecewise linear basis functions, but requires mesh construction and integration |
| Parallel computing | Supports grid-based node-wise parallel computation | Supports independent element-wise computation, followed by global assembly |
| Solution regularity | Requires higher spatial regularity | Accommodates solutions with limited spatial regularity |
| Numerical stability | Depends on the discretization and time-stepping scheme | Depends on the discretization and time-stepping scheme; may require additional stabilization |

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
h_\theta=R\cos\phi \Delta\theta,
\qquad
h_\phi=R\Delta\phi.
$$

Therefore, spherical geometry must be taken into account when evaluating spatial derivatives and surface integrals.

### 5.2 Triangular Mesh

Each rectangular cell in the latitude–longitude grid is divided into two triangular elements.

Let the four vertices of a rectangular cell be

```math
\begin{aligned}
A&=(\theta_i,\phi_j),\\
B&=(\theta_{i+1},\phi_j),\\
C&=(\theta_i,\phi_{j+1}),\\
D&=(\theta_{i+1},\phi_{j+1}).
\end{aligned}
```

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
\int_{\Omega_h} f dA
=
\sum_{e=1}^{N_e}
\int_{T_e} f dA.
$$

This allows element-level contributions to be calculated independently before being assembled into the global system.

### 5.3 Triangular Gaussian Quadrature

#### 5.3.1 General Formulation

Gaussian quadrature approximates an integral over a triangular domain $T$ using a weighted sum of function values at selected quadrature points:

$$
\int_T f(\mathbf{x}) d\mathbf{x}
\approx
\lvert T\rvert\sum_{q=1}^{Q}w_qf(\mathbf{x}_q),
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

In this model, a three-point quadrature rule is used, with quadrature points located at the midpoints of the three edges:

```math
\begin{aligned}
\mathbf{x}_{AB}&=\frac{A+B}{2},\\
\mathbf{x}_{BC}&=\frac{B+C}{2},\\
\mathbf{x}_{CA}&=\frac{C+A}{2}.
\end{aligned}
```

The corresponding weights are

$$
w_{AB}=w_{BC}=w_{CA}=\frac{1}{3}.
$$

Therefore,

$$
\int_T f(\mathbf{x}) d\mathbf{x}
\approx
\frac{\lvert T\rvert}{3}
\left[
f(\mathbf{x}_{AB})
+f(\mathbf{x}_{BC})
+f(\mathbf{x}_{CA})
\right].
$$

This quadrature rule is exact for polynomials of total degree at most two.

#### 5.3.2 Quadrature on the Spherical Surface

Let $T$ be a triangular element on the spherical surface and $\widehat{T}$ its corresponding triangle in the longitude–latitude parameter space.

Using the spherical surface area element

$$
dA=R^2\cos\phi d\theta d\phi,
$$

the surface integral can be expressed as

$$
\int_T f dA
=
\int_{\widehat{T}}
f(\theta,\phi)R^2\cos\phi d\theta d\phi.
$$

Applying the triangular Gaussian quadrature rule in the parameter space gives

$$
\int_T f dA
\approx
R^2\lvert\widehat{T}\rvert
\sum_{q=1}^{Q}
w_q f(\theta_q,\phi_q)\cos\phi_q.
$$

For the three-point quadrature rule,

$$
\int_T f dA
\approx
\frac{R^2\lvert\widehat{T}\rvert}{3}
\left[
f_{AB}\cos\phi_{AB}
+f_{BC}\cos\phi_{BC}
+f_{CA}\cos\phi_{CA}
\right],
$$

where $f_{AB}$, $f_{BC}$, and $f_{CA}$ denote the function values at the corresponding edge midpoints.

#### 5.3.3 Application to the Triangular Mesh

Consider the triangular element with vertices

```math
\begin{aligned}
A&=(\theta_i,\phi_j),\\
B&=(\theta_i+\Delta\theta,\phi_j),\\
C&=(\theta_i,\phi_j-\Delta\phi).
\end{aligned}
```

The area of the triangle in the parameter space is

$$
\lvert\widehat{T}\rvert=\frac{\Delta\theta\Delta\phi}{2}.
$$

The three quadrature points are

```math
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
```

Substituting these quadrature points and the parameter-space area into the spherical quadrature formula yields

$$
\int_T f dA
\approx
\frac{R^2\Delta\theta\Delta\phi}{6}
\left[
f_{AB}\cos\phi_j
+(f_{BC}+f_{CA})\cos\left(\phi_j-\frac{\Delta\phi}{2}\right)
\right].
$$

That formula is for the element with its base on $\phi_j$ and its third vertex to the south. On the other element the base $AB$ lies on $\phi_j-\Delta\phi$, and the third vertex lies one grid step north of $B$. The same weights then give

$$
\int_T f dA
\approx
\frac{R^2\Delta\theta\Delta\phi}{6}
\left[
f_{AB}\cos(\phi_j-\Delta\phi)
+(f_{BC}+f_{CA})\cos\left(\phi_j-\frac{\Delta\phi}{2}\right)
\right].
$$

### 5.4 Element-Level Weak Formulation

Using the conforming triangular mesh introduced above, the weak formulation can be decomposed into a sum of element-level integrals:

$$
\sum_{e=1}^{N_e}
\left[
\int_{T_e}v\frac{\partial C}{\partial t} dA
+\int_{T_e}v(\mathbf{u}\cdot\nabla_S C) dA
+\int_{T_e}K\nabla_S v\cdot\nabla_S C dA
-\int_{T_e}v(L-\lambda C) dA
\right]
=0.
$$

Each element-level integral is evaluated using local basis functions and the triangular quadrature rule introduced in Section 5.3.

The following sections describe the discretization of each term over a single triangular element. On every element, $A$ is the western vertex of the constant-latitude edge, $B$ is the eastern vertex of that edge, and $C$ is the remaining vertex. For $\triangle ABC$ in Section 5.2, $C$ is one grid step south of $A$. For $\triangle BCD$, the same labels put $A$ and $B$ on $\phi_j-\Delta\phi$ and $C$ one grid step north of $B$.

A triangle whose base lies on the north or south pole is omitted from the assembled mass and transport matrices. The diffusion weight divides by $\cos\phi$ on the base edge, and $\cos\phi=0$ at the poles.

### 5.5 Element Diffusion Matrix

From the weak formulation, the element diffusion matrix is defined as

$$
D_{ij}^{(T)}
=
\int_T K\nabla_S N_i\cdot\nabla_S N_j dA,
\qquad i,j\in\{A,B,C\}.
$$

Substituting the spherical gradient and surface area element gives

$$
D_{ij}^{(T)}
=
\int_{\widehat T}
K
\left(
\frac{1}{\cos\phi}
\frac{\partial N_i}{\partial\theta}
\frac{\partial N_j}{\partial\theta}
+
\cos\phi
\frac{\partial N_i}{\partial\phi}
\frac{\partial N_j}{\partial\phi}
\right)
d\theta d\phi.
$$

For the element with $C$ one grid step south of $A$, the P1 derivatives satisfy $\partial_\theta N_C=0$ and $\partial_\phi N_B=0$. The products of these derivatives are

```math
\left[
\partial_\theta N_i\partial_\theta N_j
\right]_{i,j}
=
\frac{1}{\Delta\theta^2}
\begin{pmatrix}
1 & -1 & 0\\
-1 & 1 & 0\\
0 & 0 & 0
\end{pmatrix},
```

```math
\left[
\partial_\phi N_i\partial_\phi N_j
\right]_{i,j}
=
\frac{1}{\Delta\phi^2}
\begin{pmatrix}
1 & 0 & -1\\
0 & 0 & 0\\
-1 & 0 & 1
\end{pmatrix}.
```

Therefore, the diffusion matrix can be decomposed into longitudinal and latitudinal contributions.

Using edge-midpoint quadrature, define

$$
\alpha_T
=
\frac{\Delta\phi}{6\Delta\theta}
\left(
\frac{K_{AB}}{\cos\phi_{AB}}
+
\frac{K_{BC}}{\cos\phi_{BC}}
+
\frac{K_{CA}}{\cos\phi_{CA}}
\right),
$$

$$
\beta_T
=
\frac{\Delta\theta}{6\Delta\phi}
\left(
K_{AB}\cos\phi_{AB}
+
K_{BC}\cos\phi_{BC}
+
K_{CA}\cos\phi_{CA}
\right),
$$

where the diffusion coefficient at each edge midpoint is approximated by the average of its endpoint values.

Since vertices A and B share the same latitude,

$$
\phi_{AB}=\phi_A=\phi_B,
\qquad
\phi_{BC}=\phi_{CA}
=\frac{\phi_A+\phi_C}{2}.
$$

When $C$ is one grid step south of $A$, edge-midpoint quadrature gives

```math
\mathbf D^{(T)}
=
\begin{pmatrix}
\alpha_T+\beta_T & -\alpha_T & -\beta_T\\
-\alpha_T & \alpha_T & 0\\
-\beta_T & 0 & \beta_T
\end{pmatrix}.
```

On the other element, $C$ is one grid step north of $B$, so $\partial_\phi N_A=0$ and $\partial_\theta N_C=0$. The same $\alpha_T$ and $\beta_T$ then give

```math
\mathbf D^{(T)}
=
\begin{pmatrix}
\alpha_T & -\alpha_T & 0\\
-\alpha_T & \alpha_T+\beta_T & -\beta_T\\
0 & -\beta_T & \beta_T
\end{pmatrix}.
```

For nonnegative $K$ and elements away from the poles, both matrices are symmetric and positive semidefinite. Their row sums are zero, so a spatially constant concentration produces no diffusion.

### 5.6 Element Advection Matrix

From the weak formulation, the element advection matrix is defined as

$$
A_{ij}^{(T)}
=
\int_T N_i(\mathbf{u}\cdot\nabla_S N_j) dA,
\qquad i,j\in\{A,B,C\}.
$$

Substituting the spherical gradient and surface area element gives

$$
A_{ij}^{(T)}
=
R\int_{\widehat T}
N_i
\left(
u\frac{\partial N_j}{\partial\theta}
+
v\cos\phi\frac{\partial N_j}{\partial\phi}
\right)
d\theta d\phi.
$$

Let $\Delta\theta>0$ and $\Delta\phi>0$ be the radian grid spacings. The longitude step is always this positive $\Delta\theta$, including from the last meridian to the first. The signed latitude step is $h_\phi=\phi_C-\phi_A$, so $h_\phi=-\Delta\phi$ when $C$ lies south of $A$ and $h_\phi=\Delta\phi$ when $C$ lies north of $B$. In the formulas below, $h_\theta=\Delta\theta$.

For the element with $C$ south of $A$, the P1 derivatives are

```math
\begin{pmatrix}
\partial_\theta N_A &
\partial_\theta N_B &
\partial_\theta N_C
\end{pmatrix}
=
\frac{1}{h_\theta}
\begin{pmatrix}
-1 & 1 & 0
\end{pmatrix},
```

```math
\begin{pmatrix}
\partial_\phi N_A &
\partial_\phi N_B &
\partial_\phi N_C
\end{pmatrix}
=
\frac{1}{h_\phi}
\begin{pmatrix}
-1 & 0 & 1
\end{pmatrix}.
```

Using edge-midpoint quadrature and the values of the P1 basis functions at the edge midpoints, define the longitudinal contribution vector

```math
\boldsymbol{\gamma}_T
=
\frac{R\Delta\theta\Delta\phi}{12h_\theta}
\begin{pmatrix}
u_{AB}+u_{CA}\\
u_{AB}+u_{BC}\\
u_{BC}+u_{CA}
\end{pmatrix},
```

and the latitudinal contribution vector

```math
\boldsymbol{\eta}_T
=
\frac{R\Delta\theta\Delta\phi}{12h_\phi}
\begin{pmatrix}
v_{AB}\cos\phi_{AB}+v_{CA}\cos\phi_{CA}\\
v_{AB}\cos\phi_{AB}+v_{BC}\cos\phi_{BC}\\
v_{BC}\cos\phi_{BC}+v_{CA}\cos\phi_{CA}
\end{pmatrix},
```

where the wind velocities at each edge midpoint are approximated by the average of their endpoint values.

When $C$ is south of $A$, the element matrix is the sum of these two contributions:

```math
\mathbf A^{(T)}
=
\boldsymbol{\gamma}_T
\begin{pmatrix}
-1 & 1 & 0
\end{pmatrix}
+
\boldsymbol{\eta}_T
\begin{pmatrix}
-1 & 0 & 1
\end{pmatrix}.
```

The resulting quadrature-based element advection matrix is

```math
\mathbf A^{(T)}
=
\begin{pmatrix}
-\gamma_A-\eta_A & \gamma_A & \eta_A\\
-\gamma_B-\eta_B & \gamma_B & \eta_B\\
-\gamma_C-\eta_C & \gamma_C & \eta_C
\end{pmatrix}.
```

On the element with $C$ north of $B$, the latitude derivatives are $(0,-1,1)/h_\phi$ instead of $(-1,0,1)/h_\phi$. With the same $\boldsymbol{\gamma}_T$ and $\boldsymbol{\eta}_T$,

```math
\mathbf A^{(T)}
=
\begin{pmatrix}
-\gamma_A & \gamma_A-\eta_A & \eta_A\\
-\gamma_B & \gamma_B-\eta_B & \eta_B\\
-\gamma_C & \gamma_C-\eta_C & \eta_C
\end{pmatrix}.
```

The advection matrix is generally nonsymmetric and depends on the wind at the current time. Both orientations have zero row sums, so a spatially constant concentration produces no advection.

## 6. Temporal Discretization

### 6.1 Semi-discrete System

After assembling the element-level mass, advection, and diffusion contributions, the weak formulation can be expressed as a semi-discrete system:

$$
M\frac{d\mathbf{C}}{dt}
+A\mathbf{C}
+D\mathbf{C}
=M(\mathbf{L}-\boldsymbol{\lambda}\odot\mathbf{C}).
$$

where:

| Notation | Description |
| --- | --- |
| $M$ | Global lumped mass matrix |
| $A$ | Global advection matrix |
| $D$ | Global diffusion matrix |
| $\mathbf{C}$ | Nodal PM2.5 concentration vector |
| $\mathbf{L}$ | Nodal source term |
| $\boldsymbol{\lambda}$ | Nodal removal coefficient |
| $\odot$ | Element-wise multiplication |

Since the lumped mass matrix is diagonal, its inverse can be applied directly through element-wise division:

$$
\frac{d\mathbf{C}}{dt}
=
-M^{-1}(A\mathbf{C}+D\mathbf{C})
+\mathbf{L}
-\boldsymbol{\lambda}\odot\mathbf{C}.
$$

### 6.2 Explicit Time Integration

Using the first-order forward Euler method, the temporal derivative is approximated as

$$
\frac{d\mathbf{C}}{dt}
\approx
\frac{\mathbf{C}_{t+1}-\mathbf{C}_t}{\Delta t}.
$$

Substituting this approximation into the semi-discrete system gives

$$
M\frac{\mathbf{C}_{t+1}-\mathbf{C}_t}{\Delta t}
+A_t\mathbf{C}_t+D_t\mathbf{C}_t
=
M(\mathbf{L}_t-\boldsymbol{\lambda}_t\odot\mathbf{C}_t).
$$

Rearranging the equation, we obtain the explicit concentration update:

```math
\mathbf{C}_{t+1}
=
\max\left(
\mathbf{C}_t
-\Delta t M^{-1}(A_t\mathbf{C}_t+D_t\mathbf{C}_t)
+\Delta t(\mathbf{L}_t-\boldsymbol{\lambda}_t\odot\mathbf{C}_t),
0
\right).
```

Negative concentrations are set to zero after the step.

### 6.3 Time Step

The observation interval and the integration step are the same:

$$
\Delta t=3\text{ hours}=10800\text{ s}.
$$

Each step uses the wind, $K$, $L$, and $\lambda$ at time $t$, advances the concentration once, and is compared with the next observation. The step is not subdivided.
