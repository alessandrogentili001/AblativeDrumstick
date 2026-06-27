# Chicken Drop From Space: Mathematical & Physical Model

This document outlines the complete mathematical and physical model implemented in the `AblativeDrumstick` project for simulating a chicken dropped from space.

---

## 1. Trajectory Dynamics & Kinematics

The vertical fall is modeled in one dimension, where **altitude decreases** and **downward velocity is treated as positive**.

### State Definition
The vertical state of the falling object at any time $t$ is represented by:
*   $h(t)$: Geometric altitude (m)
*   $v(t)$: Downward vertical velocity (m/s)

This is represented in the codebase by the [State](file:///c:/Users/aless/OneDrive/Desktop/CODING%20PROJECTS/AblativeDrumstick/src/chicken_from_space/physics.py#L9) named tuple.

### Governing Differential Equations
The system of ordinary differential equations (ODEs) integrated over time is:
$$\frac{dh}{dt} = -v$$
$$\frac{dv}{dt} = g - a_{\text{drag}}$$

where:
*   $g$: Gravity at altitude ($9.8067 \text{ m/s}^2$)
*   $a_{\text{drag}}$: Downward-convention drag acceleration ($\text{m/s}^2$)

These derivatives are calculated in the [state_derivative](file:///c:/Users/aless/OneDrive/Desktop/CODING%20PROJECTS/AblativeDrumstick/src/chicken_from_space/physics.py#L104) function.

### Aerodynamic Drag Force
The drag force magnitude is calculated assuming quadratic drag:
$$F_{\text{drag}} = \frac{1}{2} \rho C_d A v^2$$
$$a_{\text{drag}} = \frac{F_{\text{drag}}}{m}$$

where:
*   $\rho$: Atmospheric density ($\text{kg/m}^3$)
*   $C_d$: Drag coefficient (modeled as a sphere, $0.47$)
*   $A$: Cross-sectional area ($\text{m}^2$) computed from equivalent radius $r$ as $A = \pi r^2$
*   $m$: Chicken mass ($1.25 \text{ kg}$)

---

## 2. Layered Atmosphere Model

The atmosphere is modeled as a set of temperature nodes based on standard atmospheric profiles, with pressure derived from hydrostatic equilibrium and density from the ideal gas law.

### Geopotential Altitude
To account for Earth's curvature, geometric altitude $h$ is mapped to geopotential altitude $H$ via [geopotential_altitude](file:///c:/Users/aless/OneDrive/Desktop/CODING%20PROJECTS/AblativeDrumstick/src/chicken_from_space/atmosphere.py#L62):
$$H = \frac{R_E \cdot h}{R_E + h}$$
where $R_E$ is the Earth radius ($6,356,766 \text{ m}$).

### Temperature Profile
The atmospheric temperature $T(h)$ in Kelvin is linearly interpolated from a set of defined altitude-temperature nodes up to $200 \text{ km}$:
$$T(h) = \text{interpolate}(h, \vec{H}_{\text{nodes}}, \vec{T}_{\text{nodes}})$$

### Hydrostatic Pressure Integration
From the hydrostatic equation $\frac{dp}{dh} = -\rho g$ and the ideal gas law, pressure $p(h)$ is integrated using trapezoidal quadrature over 160 subdivisions:
$$p(h) = p_0 \exp\left( -\int_{0}^{h} \frac{M_{\text{air}} g}{R T(z)} dz \right)$$

where:
*   $p_0$: Sea-level pressure ($101,325 \text{ Pa}$)
*   $M_{\text{air}}$: Molar mass of air ($0.0289644 \text{ kg/mol}$)
*   $R$: Universal gas constant ($8.31446 \text{ J/(mol K)}$)

### Air Density
Density is computed using the ideal gas law:
$$\rho(h) = \frac{p(h) \cdot M_{\text{air}}}{R \cdot T(h)}$$

---

## 3. Aerothermal Boundary-Layer Heating

As the chicken falls at high speeds, compressional heating of the air boundary layer (aerothermal heating) becomes the dominant source of heat transfer.

### Speed of Sound & Mach Number
The local speed of sound $c$ and Mach number $M$ are:
$$c = \sqrt{\gamma R_{\text{spec}} T_{\text{ambient}}}$$
$$M = \frac{v}{c}$$

where:
*   $\gamma$: Ratio of specific heats for air ($1.4$)
*   $R_{\text{spec}}$: Specific gas constant for air ($287.05 \text{ J/(kg K)}$)

### Recovery (Stagnation) Temperature
The recovery temperature $T_{\text{recovery}}$ represents the fluid temperature in the boundary layer after deceleration:
$$T_{\text{recovery}} = T_{\text{ambient}} \left( 1 + r_f \frac{\gamma - 1}{2} M^2 \right)$$
where $r_f = 0.84$ is the recovery factor.

### Air Transport Properties (Sutherland's Law)
Dynamic viscosity $\mu$ is computed via Sutherland's law:
$$\mu(T) = \mu_0 \frac{ (T/T_0)^{1.5} (T_0 + S) }{ T + S }$$
where $\mu_0 = 1.716 \times 10^{-5} \text{ Pa s}$, $T_0 = 273.15 \text{ K}$, and $S = 110.4 \text{ K}$.

Thermal conductivity of air $k_f$ is:
$$k_f = \frac{\mu \cdot C_{p, \text{air}}}{Pr}$$
where $C_{p, \text{air}} = 1005 \text{ J/(kg K)}$ and Prandtl number $Pr = 0.71$.

### Convective Heat Transfer Coefficient (Ranz-Marshall Correlation)
For flow around a sphere, the Nusselt number $Nu$ and heat transfer coefficient $h_c$ are:
$$Re = \frac{\rho v D}{\mu}$$
$$Nu = 2.0 + 0.6 \sqrt{Re} Pr^{1/3}$$
$$h_c = \frac{Nu \cdot k_f}{D}$$
where $D = 2r$ is the equivalent diameter.

### Net Surface Heat Flux
The net heat flux $q_{\text{surf}}$ ($\text{W/m}^2$) entering the chicken surface accounts for convection and radiation:
$$q_{\text{surf}} = h_c (T_{\text{recovery}} - T_{\text{surface}}) - \epsilon \sigma (T_{\text{surface}}^4 - T_{\text{ambient}}^4)$$
where:
*   $\epsilon$: Emissivity of the chicken ($0.95$)
*   $\sigma$: Stefan-Boltzmann constant ($5.67037 \times 10^{-8} \text{ W/(m}^2\text{K}^4\text{)}$)

---

## 4. Internal Heat Conduction Model

The chicken is modeled as a solid sphere of radius $R_{\text{max}}$. Temperature propagation within the sphere is governed by the 1D spherical heat conduction equation:

$$\frac{\partial T}{\partial t} = \alpha_c \left( \frac{\partial^2 T}{\partial r^2} + \frac{2}{r}\frac{\partial T}{\partial r} \right)$$

where $\alpha_c = \frac{k_c}{\rho_c C_{p, c}}$ is the thermal diffusivity of chicken meat.

### Spatial Discretization
The sphere is discretized into $N$ concentric nodes ($i = 0$ at the center, $i = N-1$ at the surface) with spacing $\Delta r = R_{\text{max}} / (N - 1)$.

#### 1. Center Node ($i = 0$)
Using L'Hôpital's rule at the boundary $r \to 0$, the spatial singularity resolves to $\lim_{r\to 0} \frac{2}{r}\frac{\partial T}{\partial r} = 2 \frac{\partial^2 T}{\partial r^2}$, yielding:
$$\frac{dT_0}{dt} = 6 \alpha_c \left( \frac{T_1 - T_0}{\Delta r^2} \right)$$

#### 2. Intermediate Nodes ($1 \le i \le N - 2$)
Using central finite differences:
$$\frac{dT_i}{dt} = \alpha_c \left[ \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta r^2} + \frac{2}{r_i} \left( \frac{T_{i+1} - T_{i-1}}{2\Delta r} \right) \right]$$
where $r_i = i \cdot \Delta r$.

#### 3. Surface Node ($i = N-1$)
Accounting for surface heat flux $q_{\text{surf}}$ boundary condition:
$$\frac{dT_{N-1}}{dt} = \frac{2}{\rho_c C_{p,c} \Delta r} \left[ q_{\text{surf}} - \left(1 - \frac{\Delta r}{2 R_{\text{max}}}\right)^2 k_c \left(\frac{T_{N-1} - T_{N-2}}{\Delta r}\right) \right]$$
