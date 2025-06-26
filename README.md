# Chaotic System Applications in Cryptography

## Background
Chaotic Systems are incredibly interesting systems, given that they have two defining features - they are *chaotic* and they are *deterministic*.

*Chaotic*, in this context, means unpredictable. I.E. the system is heavily dependent on its input conditions. In such as situation, if you caclcuated the result at a time *t* from systems with similar (even similar on the order of a 1%) input conditions, the resulting values of *x*, *y*, *z* returned by the system would be wildly different.

*Deterministic* means that while the system is unpredictable for us, if you gave two systems identical input parameters, they would give you indentical results.

For the purposes of this project, and the background, we are going to be focused on one system of differential equations and it's chaotic soltuions - the Lorenz System of Differential Equations.
This is arguably one of the most famous or impactful systems with chaotic solutions. Many of you have probabaly heard of the Butterfly Effect. In fact, this popular phenomena may be the result of this system and the implications of its chaotic solutions. It probabaly also helps that this system's chaotic solutions, when mapped in 3D, sort of resembles a butterfly.

Lorenz and co. originally developed this system to map or track atmospheric convection, which could help with weather prediction. However, this system appears all over simiplified models for other utilities, such as lasers, electical circuits, chemical reactions and many more. If you find this interesting, I highly reccomend researching the topic more.

The System looks as follows:
- dx / dt = σ(y − x)
- dy / dt = rx − y − xz
- dz / dt = xy − b


The Lorenz System, when mapped in 3D with the chaoatic solution of σ=10, b=8/3 and r=28, is displayed below.

<img src="system_plots/r_values/r=28/Figure 2024-08-18 193846.png" alt="A 3D plot with a line that travels in two interlocking circular patterns, but does not touch any other line.">


With different r-values, of course, the system isn't chaotic, as demonstrated by this graph with conditions σ=10, b=8/3 and r=10. The system gradually approaches its singualr solution.

<img src="system_plots/r_values/r=10/Figure 2024-08-18 193426.png" alt="A 3D plot with a line that travels in a signal spiralling circular pattern toward a singalular point.">

It is far easier to see in this 2D graph, where the *(x, y, z)* coordinates are plotted against time *t*. All three coordinates approach a single value (seen on the left). Conversely, as you can see with the true chaotic solution, the values continue to alternate widly, with not truly repeated values (seen on the right).

<img src="system_plots/r_values/r=10/Figure 2024-08-18 193459.png" alt="A 2D plot three lines that begin by alternating wildly but then converge to single values."><img src="system_plots/r_values/r=28/Figure 2024-08-18 193905.png" alt="A 2D plot three lines that alternate wildly.">

Interesting, there are also values that are semi-choatic. That is, at small values of *t*, the system is chaotic, before later converging at larger values of *t*.

<img src="system_plots/r_values/r=20/Figure 2024-08-18 193625.png" alt="A 2D plot three lines that begin by alternating wildly but then converge to single values.">

## Integration
This project is not necessarily only for those who understand differntial equations - if you are familiar with differential equations, skip the following section.

Put simply (and sort of erroneously) a derivitive is an equation that describes the rate-of-change of another equation. If you mapped an object's location over time with an equation, then the derivitive of that equation would map that object's speed over time. For example, if your driving your car at 35mph, that equation would be x (distance traveled in miles) = 35 (miles-per-hour) * t (time travelled in hours). The deriviative of that equation is then dx/dt (change in miles / time in hours) = 35 (miles-per-hour). This is because your speed is a constant 35 miles-per-hour. 
Now, a differential equation is simply an equation that maps some function or unknown quantity to its derivitive. In theory, these sound simple, but they can quickly become complicated and difficult to solve (as is the case with the Lorenz system). As such, we use an *intergrator*. An integrator is merely a tool that uses mathematics to guess (quite accurately) what the value of *x* is at any given point based on its derivitve and time *t*. In our example of dx/dt = 35mph, if we know that *t* = 1 hour, then we know we have travelled 35 miles. As such, we solved *x* solely based on dx / dt and t,
Now, we solved that analytically, but for more complicated equations, such as the Lorenz system, it is merely too time consuming or too diffult to solve it analytically. Instead we use an integrator. An integrator basically uses a lot of high-fidelity and precise approximations to accurately guess what the value of *x* is based off of *dx/dt* and *t*.

The integrator we are using for this project is an RK-45 integrator. An RK-45 integrator is an integrator based on a mathametical integration technique developed by Carl Runge and Wilhelm Kutta. This particular integrator uses adaptive step-sizing to reduce the compute time when integrating.

All of the above graphs were mapped by using this integrator to get the values of *(x, y, z)* and mapping them at the given time *t*.

If you would live to try this for yourself, you are welcome to. You can observe the output of the Lorenze system of differntial equations at many different values by downloading the [exploration module](exploration.py). You can pass in different sigma values, different r values, and different b values by using the *--sig*, *--b* and *--r-values* command-line-arguments.

## Communicating with Chaos
A Choatic Systems best feature is just that - it's non-periodic chaotic solutions. These chaotic solutions can have some integresting applications - specifically, in this study, for cryptography.

Because the results of chaotic systems are incredibly dependent on initial solutions, the same chaptic system will return wildly-varying non-periodic unique results. 

Another interesting result is that systems can synchronized. For example, lets say we have persons A and B. They do not know the initial conditions of the each others systems, however, they *do* know that they both have a Lorenz system, and they agree on the following: person A will transmit their result when they integrate dx/dt. Now, during normal mathematical integration, the integrator will use the function's previously calculated value for *x* to determine the next value of *x*.

Person A will send Person B all the values they got for *x* when they integrated it.  Person B, instead of using the value for x that they calulated will instead use the value that Person A transmitted to them. The two systems will eventually syncrhonize. This can be demonstrated in the [communication module](communicaton.py) by using the *--check-for-synchronization-conditions* command-line-argument. For the equation (the one representing person A) it will either use the defaults values for r, b, sigma and the intial conditions, or the ones provided by the user. 
For the second system, the same r, b, and sigma values are used, but the inital conditions are different and randomly generated. As you can see below, the two systems eventually syncrhonize. 
*(x, y, z)* refer to the coordinates of the original system (or the system beloning to person A, and *(u, v, w)* refer to the coordinates of the system with randomly generated initial conditions, or the system belonging to person B, and the system synchronizing to person A.
The individual values of x, y, z and u, v, w are mapped below.

<img src="system_plots/synchronization/Figure 2025-06-26 133814.png" alt="A 2D plot mapping the x and u."><img src="system_plots/synchronization/Figure 2025-06-26 133845.png" alt="A 2D plot mapping y and v."><img src="system_plots/synchronization/Figure 2025-06-26 133849.png" alt="A 2D plot mapping the difference between z and w.">


As you can see, the two systems synchronize and produce the same result. This can be mroe clearly seen if you plot the differences between x and u, y and v and z and w. These are mapped belowby the absolute values between their coodrinates (labelled as |x-u|, |y-v| and |z-w|).

<img src="system_plots/synchronization/Figure 2025-06-26 133857.png" alt="A 2D plot mapping the difference between x and u."><img src="system_plots/synchronization/Figure 2025-06-26 133903.png" alt="A 2D plot mapping the difference between y and v."><img src="system_plots/synchronization/Figure 2025-06-26 133906.png" alt="A 2D plot mapping the difference between z and w.">

We can take advatage of this if we wanted to send an encrypted wave, for example, a sound file.

To do this, there are a few steps. 
- Person A integrates the system as normal. Person B knows the system, but not its initial consditions. They agree on the variable they are syncrhonizing on. In this case, they are syncrhonizing the result of *x*.
  - Normally, and RK45 integrator has variable step-sizing. In order to maintain the fidelity of the sound file, they have both also agreed to use fixed, agreed-upon step-sizes.
- Person A takes the wave they want to transmit, and minimizes it, multplying it by a factor of 0.001 or 0.0001. This means the amplitude of the wave is now about 1% or 0.1% the peaks of the chaotic distribution. 
- Person A then perterbs the choatic distribution, or, adds the minimized sound wave to the chaotic function of the variable of their choice (in this case, the result of the integration for the variable *x*).
- The result is then transmitted to Person B.
- Person B synchronizes their system to the transmitted wave.
- Person B then subtracts their result from the transmitted signal.
- The result is then divided by the normalization factor to get the original wave.








