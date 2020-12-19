# Importing relevant modules
from download_data import data_to_csv  # Importing method to download data using shortcode
import plots  # Plotting module
from nll import NLL  # Module which generates the oscillated event rate
from minim_1d import Minimise1D  # Parabolic minimisation module
from minim_2d import Minimise2D  # 2-D minimisation module
from minim_3d import Minimise3D  # 3-D minimisation module

# Importing relevant packages
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# # Defining colour list (to use in plots)
colour_list = list(mcolors.TABLEAU_COLORS.values())  # Using the Tableau colours from matplotlib.colors

"""
Section 3.1 - Generating Personalised Data and Preliminary Analysis
"""
shortcode = 'wyw18'
# data_to_csv(shortcode)  # Generate and download data file from shortcode

# Loading the data that was saved as a .csv file
data = np.loadtxt(f'{shortcode}_data.csv', delimiter = ',', skiprows = 1).T  # Transpose to load the data saved in column form
exp_data = np.array(data[0])
event_no = np.array(data[1])

# # Plotting histograms of the data over the range 0-10 GHz, with varying numbers of bins (and thus varying intervals)
# bins_list = [200, 100, 50, 25]
# for i, val in enumerate(bins_list):
#     plots.histogram(exp_data, val, f"hist_{val}", f"Plot of Experimental Data ({val} bins)", xlabel = 'Energy (GeV)', ylabel = 'Frequency', color = colour_list[i])

# plots.histogram(event_no, 200, "hist_event", f"Plot of Simulated Event Rate (Number)", xlabel = 'Energy (GeV)', ylabel = 'Simulated Event Rate', color = '#9403fc')

"""
Section 3.2 - Calculating Probabilities and Investigating Oscillation Parameters
"""
en_array = np.linspace(0, 10, num = 200, endpoint = False) + 0.025  # Creating an array of energies - i.e. the midpoints of the energy bins
# Defining default values for oscillation parameters
mix_ang = np.pi / 4
L = 295
sq_mass_diff = 2.4e-3

# # Creating an NLL object in order to calculate the survival probabilities of a muon neutrino at different energies
# nll_obj = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = mix_ang, distance = L, sq_mass_diff = sq_mass_diff)
# # Finding the survival probability (i.e. the probability that ν_µ will not change into a neutrino of a different flavour)
# p = nll_obj.surv_prob() 

# plots.plot(en_array, p, filename = 'energy_plot', title = r"Plot of P($\nu_\mu \rightarrow \nu_\mu$) against E", \
#             xlabel = "Energy (GeV)", ylabel= "Survival Probability", clear = True, color = "#b103fc")


# """
# Creating surface plots for visualisation (in our case the z-axis is the probability)
# """
# # Varying the mixing angle and generating a surface plot 
# ang_iters = 25  # Number of iterations
# mix_ang_arr = np.linspace(0, np.pi, ang_iters)
# ang_coords = np.empty(shape = (200 * ang_iters, 3))  # Creating a placeholder coordinate array, with columns [x,y,z]

# for ind, val in enumerate(mix_ang_arr):
#     nll_obj_t = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = val, distance = L, sq_mass_diff = sq_mass_diff)
#     prob_t = nll_obj_t.surv_prob()
#     angles = [val] * 200  # Mixing angle coordinate (y)
    
#    # Assigning coordinates to coordinate array
#     start_ind = ind * 200  # Start row index
#     end_ind = (ind + 1) * 200  # End row index

#     ang_coords[start_ind:end_ind, 0] = en_array  # Assigning x-coordinates
#     ang_coords[start_ind:end_ind, 1] = angles  # Assigning y-coordinates
#     ang_coords[start_ind:end_ind, 2] = prob_t  # Assigning y-coordinates

# plots.surf_plot(ang_coords[:,0], ang_coords[:,1], ang_coords[:,2], filename = "surf_plot_mix_ang", \
#                 title = r"Surface plot of P($\nu_\mu \rightarrow \nu_\mu$) against E and $\theta_{23}$", \
#                 xlabel = "Energy (GeV)", ylabel = "Mixing Angle", zlabel = "Survival Probablility", elev = 8.323, azim = -41)


# # Varying the squared mass difference and generating a surface plot 
# mass_iters = 25  # Number of iterations
# sq_mass_arr = np.linspace(1e-4, 1e-2, mass_iters)
# mass_coords = np.empty(shape = (200 * mass_iters, 3))  # Creating a placeholder coordinate array, with columns [x,y,z]

# for ind, val in enumerate(sq_mass_arr):
#     nll_obj_m = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = mix_ang, distance = L, sq_mass_diff = val)
#     prob_m = nll_obj_m.surv_prob()
#     masses = [val] * 200  # Mixing angle coordinate (y)
    
#     # Assigning coordinates to coordinate array
#     start_ind = ind * 200  # Start row index
#     end_ind = (ind + 1) * 200  # End row index

#     mass_coords[start_ind:end_ind, 0] = en_array  # Assigning x-coordinates
#     mass_coords[start_ind:end_ind, 1] = masses  # Assigning y-coordinates
#     mass_coords[start_ind:end_ind, 2] = prob_m  # Assigning y-coordinates

# # Generating and saving a plot of P vs E and squared mass difference
# plots.surf_plot(mass_coords[:,0], mass_coords[:,1], mass_coords[:,2], filename = "surf_plot_sq_mass", \
#                 title = r"Surface plot of P($\nu_\mu \rightarrow \nu_\mu$) against E and $\Delta_{23}^2$", \
#                 xlabel = "Energy (GeV)", ylabel = "Squared Mass Difference", zlabel = "Survival Probablility", elev = 28.7, azim = 71.6)

# # Varying the distance (L) and generating a surface plot 
# dist_iters = 25  # Number of iterations
# dist_arr = np.linspace(0, 2500, dist_iters)
# dist_coords = np.empty(shape = (200 * dist_iters, 3))  # Creating a placeholder coordinate array, with columns [x,y,z]

# for ind, val in enumerate(dist_arr):
#     nll_obj_d = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = mix_ang, distance = val, sq_mass_diff = sq_mass_diff)
#     prob_d = nll_obj_d.surv_prob()
#     dists = [val] * 200  # Mixing angle coordinate (y)
    
#    # Assigning coordinates to coordinate array
#     start_ind = ind * 200  # Start row index
#     end_ind = (ind + 1) * 200  # End row index

#     dist_coords[start_ind:end_ind, 0] = en_array  # Assigning x-coordinates
#     dist_coords[start_ind:end_ind, 1] = dists  # Assigning y-coordinates
#     dist_coords[start_ind:end_ind, 2] = prob_d  # Assigning y-coordinates

# plots.surf_plot(dist_coords[:,0], dist_coords[:,1], dist_coords[:,2], filename = "surf_plot_dist", \
#                 title = r"Surface plot of P($\nu_\mu \rightarrow \nu_\mu$) against E and L", \
#                 xlabel = "Energy (GeV)", ylabel = "Distance (km)", zlabel = "Survival Probablility", elev = 25, azim = 68)

# """
# Section 3.3 - Calculating and plotting the Negative Log Likelihood (NLL)
# """
# num_thetas = 500  # Number of mixing angles (θ_23) to investigate
# theta_arr = np.linspace(0, 2 * np.pi, num_thetas)  # Generating an array of mixing angles from [0,2π]

# # Note that initialising an empty NumPy array and then assigning values is faster than appending
# # --> as the interpreter has to assign memory with every appending operation.
# nll_vals = np.empty(num_thetas)  # Creating an empty NumPy array for the NLL values

# # Finding the NLL for each mixing angle in theta_arr
# for ind, val in enumerate(theta_arr):
#     # Creating instance of NLL class, and finding lists of P and λ
#     nll_theta = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = val, distance = L, sq_mass_diff = sq_mass_diff)
#     nll_theta.surv_prob()
#     nll_theta.calc_lambda()

#     nll = nll_theta.find_nll()  # Calculating the NLL
#     nll_vals[ind] = nll  # Assigns the calculated NLL value with the corresponding index

# # Generating and saving a plot of NLL vs mixing angle
# plots.plot(theta_arr, nll_vals, filename = "nll_thetas", title = r"Plot of NLL against $\theta_{23}$", \
#             xlabel = r"$\theta_{23}$ (rads)", ylabel = "NLL", color = '#249c00')

"""
Section 3.4 - Parabolic Minimisation
- From inspection of the NLL function plotted in Section 3.3 (in nll_thetas.pdf), it can be seen that the function has period π/2, and that
  the minima are periodic. The first minimum region was chosen to minimise over, the numerical range of which was chosen by inspection.
- As there are two local minima per period, the parabolic minimisation algorithm is applied twice in order to investigate the behaviour further.
"""
# As the NLL function has period π/2, we take the first period to investigate it in more detail:
data = [en_array, event_no, exp_data]  # Data to be passed into the Minimise1D object

# # Investigating the left local minimum:
# # Creating a Minimise1D object for parabolic minimisation
# min_obj = Minimise1D(init_range = [0.55, 0.78], nll = True, nll_param = 'theta', nll_data = data)  # The first range of [0.55,0.78] was chosen by inspection of the previous plot
# min_theta = min_obj.para_min()
# print("--- Left Minimum ---")
# print(f"Mixing Angle which minimises NLL: {min_theta}")
# print(f"NLL value: {min_obj.min_func}")
# print(f"Iterations: {min_obj.iterations}")

# # Investigating the right local minimum:
# # Creating a Minimise1D object for parabolic minimisation
# min_obj2 = Minimise1D(init_range = [0.795, 1.02], nll = True, nll_param = 'theta', nll_data = data)  # The second range of [0.795,1.02] was chosen by inspection of the previous plot
# min_theta2 = min_obj2.para_min()
# print("--- Right Minimum ---")
# print(f"Mixing Angle which minimises NLL: {min_theta2}")
# print(f"NLL value: {min_obj2.min_func}")
# print(f"Iterations: {min_obj2.iterations}")

# # What if range includes both local minima? --> don't need
# min_obj3 = Minimise1D(init_range = [0.2, 1.2], nll = True, nll_param = 'theta', nll_data = data)
# min_theta3 = min_obj3.para_min()
# print("--- Minimising over both local minima ---")
# print(f"Mixing Angle which minimises NLL: {min_theta3}")
# print(f"NLL value: {min_obj3.min_func}")
# print(f"Iterations: {min_obj3.iterations}")

# # Testing the minimisation on an arbitrary function:
# def parabola(x):
#   return x ** 2 - 2 * x + 1  # Parabola with minimum at (1,0)
# # Creating a Minimise1D object for parabolic minimisation
# min_obj4 = Minimise1D(init_range = [-5, 5], nll = False, nll_param = None, nll_data = None, func = parabola)
# min_x = min_obj4.para_min()
# print("--- Parabola Function: y = x^2 - 2x + 1 ---")
# print(f"x-value which minimises f(x): {min_x}")
# print(f"y-value: {min_obj4.min_func}")
# print(f"Iterations: {min_obj4.iterations}")

# """
# Section 3.5 - Accuracy of Parabolic Fit
# - An estimate of the standard deviation of the mixing angle is then found using the change in the NLL.
# - A second estimate is then found by looking at the curvature of the NLL around the minimum:
#   --> i.e. the negative log likelihood can be approximated as an inverted Gaussian around the minimum.
#   --> This allows us to obtain a reasonable estimate for σ.
# """
# # Finding standard deviation (and corresponding error measurements) using the change in the NLL:
# std_arr = min_obj2.std_change(return_all = True)  # Using the same Minimise1D object as for the right local minimum (of the first period)
# print("--- Calculating Standard Deviation (Right Local Minimum) ---")
# print("--> Using Change in NLL:")
# print(f"Standard deviation of θ: {std_arr[0]}")
# print(f"θ+, θ-: ({std_arr[1]}, {std_arr[2]})")
# print(f"Corresponding NLL values: ({std_arr[3]}, {std_arr[4]})")

# # Finding standard deviarion using Gaussian approximation:
# print("--> Using Gaussian Approximation:")
# print(f"Standard deviation of θ: {min_obj2.std_gauss()}")

# """
# Section 4 - Preliminary Investigations
# - Using values around the minimising value of theta, the behaviour of the NLL with varying squared mass difference is then investigated.
# - Also created contour plots to further investigate the behaviour.
# """
# num_m = 500  # Number of squared mass differences to investigate
# m_arr = np.linspace(1e-4, 1e-2, num_m)  # Generating an array of squared mass differences from [10^-4,10^-2]
# thetas = [min_theta2 - 0.15, min_theta2, min_theta2 + 0.15, min_theta2 + 0.3]
# plt.clf()  # Clears previous figures
# plot_counter = 0  # Plot counter
# for theta in thetas:
#   nll_m = np.empty(num_m)
#   # Finding the NLL for each squared mass diff
#   for ind, val in enumerate(m_arr):
#       # Creating instance of NLL class, and finding lists of P and λ
#       nll_theta = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = theta, distance = L, sq_mass_diff = val)
#       nll_theta.surv_prob()
#       nll_theta.calc_lambda()

#       nll = nll_theta.find_nll()  # Calculating the NLL
#       nll_m[ind] = nll  # Assigns the calculated NLL value with the corresponding index

#   if plot_counter < len(thetas) - 1:  # If function plotted is not the last
#     # Generating (but not saving) a plot of NLL vs squared mass diff
#     plots.plot(m_arr, nll_m, filename = "nll_sqmass", title = r"Plot of NLL against $\Delta_{23}^2$", \
#                 xlabel = r"$\Delta_{23}^2$ (eV$^2$)", ylabel = "NLL", clear = False, save = False, label = f"$θ_{{{23}}} = {round(theta, 2)}$")
#   else:
#     # Saving the plot upon plotting the last NLL function
#     plots.plot(m_arr, nll_m, filename = "nll_sqmass", title = r"Plot of NLL against $\Delta_{23}^2$", \
#                 xlabel = r"$\Delta_{23}^2$ (eV$^2$)", ylabel = "NLL", clear = False, legend = True, save = True, label = f"$θ_{{{23}}} = {round(theta, 2)}$")
  
#   plot_counter += 1  # Increments counter by 1

# # Creating contour plots
# x_theta = np.linspace(0.3, 1.2, 50)  # Array of mixing angles
# y_mass = np.linspace(5e-4, 5e-3, 50)  # Array of squared mass differences
# X, Y = np.meshgrid(x_theta, y_mass)  # Forms two 'grids' of X and Y data
# z_nll = np.empty((50,50))  # Empty 2-D array initialised for the NLL values
# # Calculating 50 x 50 NLL values for the contour plot
# for i in range(0, len(x_theta)):
#   for j in range(0, len(y_mass)):
#     nll_contour = NLL(en_array, event_no, exp_data, mix_ang = X[i][j], distance = L, sq_mass_diff = Y[i][j])
#     nll_contour.surv_prob()
#     nll_contour.calc_lambda()
#     z_nll[i][j] = nll_contour.find_nll()  # Calculating the NLL
# # Contour line plot
# plots.contour(X, Y, z_nll, filename = "contour", colorbar = False, fill = False, \
#               title = "Contour Plot of NLL vs Mixing Angle and squared mass difference", xlabel = r"$\theta_{23}$ (rads)", ylabel = r"$\Delta_{23}^2$ (eV$^2$)")
# # Filled contour plot
# plots.contour(X, Y, z_nll, filename = "contour_fill", colorbar = True, fill = True, \
#               title = "Contour Plot of NLL vs Mixing Angle and squared mass difference", xlabel = r"$\theta_{23}$ (rads)", ylabel = r"$\Delta_{23}^2$ (eV$^2$)")

"""
Section 4.1 - Univariate 2-D Minimisation
- Note: x refers to the mixing angle, and y refers to the squared mass difference.
"""
data = [en_array, event_no, exp_data]  # Data to be passed into the Minimise2D object
# # Creating a Minimise2D object for univariate minimisation (also used later)
# min_2d = Minimise2D([0.55, 0.78], [1e-3, 4e-3], nll = True, nll_data = data, start_coord = [0.57, 1.5e-3])
# # Minimising the mixing angle (x-direction) first
# min_2d.univ_min(first = 'x')
# print("--- 2-D Univariate Minimisation (Mixing Angle first) ---")
# print(f"Mixing Angle which minimises NLL: {min_2d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_2d.min[1]}")
# print(f"NLL value: {min_2d.dir_min_func}")
# print(f"Total iterations: {min_2d.iterations}")
# print(f"x-direction --> Iterations: {min_2d.x_iters}, Minimisations: {min_2d.min_iters_x}")
# print(f"y-direction --> Iterations: {min_2d.y_iters}, Minimisations: {min_2d.min_iters_y}")
# univ_mins = min_2d.mins_list

# # Minimising the squared mass difference (y-direction) first
# min_2d.univ_min(first = 'y')
# print("--- 2-D Univariate Minimisation (Squared Mass Diff first)---")
# print(f"Mixing Angle which minimises NLL: {min_2d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_2d.min[1]}")
# print(f"NLL value: {min_2d.dir_min_func}")
# print(f"Total iterations: {min_2d.iterations}")
# print(f"x-direction --> Iterations: {min_2d.x_iters}, Minimisations: {min_2d.min_iters_x}")
# print(f"y-direction --> Iterations: {min_2d.y_iters}, Minimisations: {min_2d.min_iters_y}")

"""
Section 4.2 - Testing Simultaneous Minimisation Schemes
"""
# # Gradient scheme - Note this takes a long time to run (~3k iterations)
# min_2d.grad_min(alpha = 2e-7)
# print("--- 2-D Simultaneous Minimisation (Gradient Method) ---")
# print(f"Mixing Angle which minimises NLL: {min_2d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_2d.min[1]}")
# print(f"NLL value: {min_2d.nll_min}")
# print(f"Total iterations: {min_2d.iterations}")
# grad_mins = min_2d.mins_list
# grad_mins = np.vstack(grad_mins)

# # Newton scheme
# min_2d.newton_min(alpha = 2e-7)
# print("--- 2-D Simultaneous Minimisation (Newton Method) ---")
# print(f"Mixing Angle which minimises NLL: {min_2d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_2d.min[1]}")
# print(f"NLL value: {min_2d.nll_min}")
# print(f"Total iterations: {min_2d.iterations}")
# newt_mins = min_2d.mins_list
# newt_mins = np.vstack(newt_mins)
# print(newt_mins[:,0])

# # Quasi-Newton scheme - Note this also takes a long time to run (~3k iterations)
# min_2d.quasi_newton_min(alpha = 2e-7)
# print("--- 2-D Simultaneous Minimisation (Quasi-Newton Method) ---")
# print(f"Mixing Angle which minimises NLL: {min_2d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_2d.min[1]}")
# print(f"NLL value: {min_2d.nll_min}")
# print(f"Total iterations: {min_2d.iterations}")
# quas_mins = min_2d.mins_list
# quas_mins = np.vstack(quas_mins)
# print(quas_mins[:,0])

# """
# Section 5 - Preliminary Investigations
# - Using the minimising values for the mixing angle and the squared mass difference, investigate
#   the effect of the neutrino interaction cross-section -i.e. its scaling with neutrino energy.
# """
# num_cross_secs = 500  # Number of cross-sections (α) to investigate
# cs_arr = np.linspace(0.1, 10, num_cross_secs)  # Generating an array of cross-sections from [0,10]

# nll_vals = np.empty(num_cross_secs)  # Creating an empty NumPy array for the NLL values

# # Finding the NLL for each mixing angle in theta_arr
# for ind, val in enumerate(cs_arr):
#     # Creating instance of NLL class, and finding lists of P and λ
#     nll_cs = NLL(energies = en_array, event_rates = event_no, obs_events = exp_data, mix_ang = 0.67299, distance = L, sq_mass_diff = 2.589e-3, cross_sec = val)
#     nll_cs.surv_prob()
#     nll_cs.calc_lambda()

#     nll = nll_cs.find_nll()  # Calculating the NLL
#     nll_vals[ind] = nll  # Assigns the calculated NLL value with the corresponding index

# # Generating and saving a plot of NLL vs mixing angle
# plots.plot(cs_arr, nll_vals, filename = "nll_cross_secs", title = "Plot of NLL against Cross Section scaling with Energy)", \
#             xlabel = "Cross Section Proportionality", ylabel = "NLL", color = '#ff089c')

# # Creating contour plot - remove
# x_theta = np.linspace(0.3, 1.2, 50)  # Array of mixing angles
# y_cs = np.linspace(0.1, 3, 50)  # Array of cross-section proportionality constants
# X, Y = np.meshgrid(x_theta, y_cs)  # Forms two 'grids' of X and Y data
# z_nll = np.empty((50,50))  # Empty 2-D array initialised for the NLL values
# # Calculating 50 x 50 NLL values for the contour plot
# for i in range(0, len(x_theta)):
#   for j in range(0, len(y_cs)):
#     nll_contour = NLL(en_array, event_no, exp_data, mix_ang = X[i][j], distance = L, sq_mass_diff = 2.71e-3, cross_sec = Y[i][j])
#     nll_contour.surv_prob()
#     nll_contour.calc_lambda()
#     z_nll[i][j] = nll_contour.find_nll()  # Calculating the NLL
# # Contour line plot
# plots.contour(X, Y, z_nll, filename = "contour_cross_sec", colorbar = False, fill = False, \
#               title = "Contour Plot of NLL vs Mixing Angle and Cross-Section Proportionality", xlabel = r"$\theta_{23}$ (rads)", ylabel = "Cross Section Proportionality")

"""
Section 5 - Testing 3-D Minimisation Schemes
- The schemes previously tested in section 4 were generalised to 3 dimensions, and tested.
- Note: x refers to the mixing angle, y refers to the squared mass difference, and z refers to the cross-section proportionality constant.
- From inspection (nll_cross_secs.pdf), the z-initialisation range was chosen to be [0.5,2].
"""
data = [en_array, event_no, exp_data]  # Data to be passed into the Minimise2D object
# Creating a Minimise3D object for univariate minimisation (also used later)
min_3d = Minimise3D([0.55, 0.78], [1e-3, 4e-3], [0.5,2], nll = True, nll_data = data)
# min_3d = Minimise3D([0.55, 0.78], [1e-3, 4e-3], [0.5,2], nll = True, nll_data = data, start_coord = [0.675, 2.5e-3, 1.5])
# Minimising the cross-section scaling (z-direction) first
min_3d.univ_min(first = 'z')
print("--- 3-D Univariate Minimisation (Mixing Angle first) ---")
print(f"Mixing Angle which minimises NLL: {min_3d.min[0]}")
print(f"Squared Mass Difference which minimises NLL: {min_3d.min[1]}")
print(f"Cross section-Energy proportionality constant which minimises NLL: {min_3d.min[2]}")
print(f"NLL value: {min_3d.dir_min_func}")
print(f"Total iterations: {min_3d.iterations}")
print(f"x-direction --> Iterations: {min_3d.x_iters}, Minimisations: {min_3d.min_iters_x}")
print(f"y-direction --> Iterations: {min_3d.y_iters}, Minimisations: {min_3d.min_iters_y}")
print(f"z-direction --> Iterations: {min_3d.z_iters}, Minimisations: {min_3d.min_iters_z}")
univ_mins = min_3d.mins_list
print(min_3d.start)
print(univ_mins[0])

# Gradient scheme - Note this takes a long time to run (~3k iterations)
# min_3d.grad_min(alpha = 1e-6)
# print("--- 3-D Simultaneous Minimisation (Gradient Method) ---")
# print(f"Mixing Angle which minimises NLL: {min_3d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_3d.min[1]}")
# print(f"Cross section-Energy proportionality constant which minimises NLL: {min_3d.min[2]}")
# print(f"NLL value: {min_3d.nll_min}")
# print(f"Total iterations: {min_3d.iterations}")
# grad_mins = min_3d.mins_list
# grad_mins = np.vstack(grad_mins)

# # Quasi-Newton scheme - Note this takes a long time to run (~1k iterations)
# min_3d.quasi_newton_min(alpha = 1e-6)
# print("--- 3-D Simultaneous Minimisation (Quasi-Newton Method) ---")
# print(f"Mixing Angle which minimises NLL: {min_3d.min[0]}")
# print(f"Squared Mass Difference which minimises NLL: {min_3d.min[1]}")
# print(f"Cross section-Energy proportionality constant which minimises NLL: {min_3d.min[2]}")
# print(f"NLL value: {min_3d.nll_min}")
# print(f"Total iterations: {min_3d.iterations}")
# quas_mins = min_3d.mins_list
# quas_mins = np.vstack(quas_mins)
# print(quas_mins[:,0])
