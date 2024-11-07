import magnetdesigner
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Configuration
    fieldTolerances = [1e-4, 1e-3, 1e-2]
    B_design = 1.5
    aper_x = 0.2
    aper_y = 0.1
    current_densities = np.linspace(0.5, 10, 1000)
    currentfactors = [40.0/125.0, 1.0]
    length = 2.0
    ylim=1.85

    # Helper function to calculate costs for each current density
    def calculate_costs(B_design, aper_x, aper_y, current_densities, fieldTolerance, currentfactor=1.0, length=2.0):
        cost_array = []
        yoke_array = []
        coil_array = []
        operation_array = []

        for current_density in current_densities:
            # Obtain magnetic design data from the designer
            bend = magnetdesigner.designer.get_Dipole(
                B_design=B_design, aper_x=aper_x, aper_y=aper_y, maxCurrentDensity=current_density, fieldTolerance=fieldTolerance, length=length
            )

            # Extract the relevant cost data
            yoke_cost = bend.magnet2D.yokeCostTotal * 1e-6
            coil_cost = bend.magnet2D.coilCostTotal * 1e-6
            operating_cost = bend.operatingCosts * 1e-6 * currentfactor
            total_cost = (yoke_cost + coil_cost + operating_cost)

            # Append the results to the arrays
            yoke_array.append(yoke_cost)
            coil_array.append(coil_cost)
            operation_array.append(operating_cost)
            cost_array.append(total_cost)

        return cost_array, yoke_array, coil_array, operation_array

    # Create cost arrays for each tolerance
    tol_array_cost = []
    tol_array_yoke = []
    tol_array_coil = []
    tol_array_operation = []

    for currentfactor in currentfactors:
        for fieldTolerance in fieldTolerances:
            costarray, yokearray, coilarray, operationarray = calculate_costs(B_design, aper_x, aper_y, current_densities, fieldTolerance, currentfactor, length)
            tol_array_cost.append(costarray)
            tol_array_yoke.append(yokearray)
            tol_array_coil.append(coilarray)
            tol_array_operation.append(operationarray)

    # Plot function
    def plot_cost(ax, current_densities, total_costs, yoke_costs, coil_costs, operation_costs):
        ax.plot(current_densities, total_costs, label='Total Cost', color='black', lw=2)
        ax.plot(current_densities, yoke_costs, label='Yoke Cost', color='darkgrey')
        ax.plot(current_densities, coil_costs, label='Coil Cost', color='brown')
        ax.plot(current_densities, operation_costs, label='Operating Cost', color='blue')

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Flatten the axes array and iterate over it with corresponding tolerance data
    for ax, costarray, yokearray, coilarray, operationarray in zip(axes.flat, tol_array_cost, tol_array_yoke, tol_array_coil, tol_array_operation):
        plot_cost(ax, current_densities, costarray, yokearray, coilarray, operationarray)
        ax.set_ylim(0, ylim)

    # Set labels and titles
    axes[0, 0].set_title('Elecricity Cost: 40 EUR/MWh\nField Tolerance: 1e-4')
    axes[0, 1].set_title('Elecricity Cost: 40 EUR/MWh\nField Tolerance: 1e-3')
    axes[0, 2].set_title('Elecricity Cost: 40 EUR/MWh\nField Tolerance: 1e-2')
    axes[1, 0].set_title('Elecricity Cost: 125 EUR/MWh\nField Tolerance: 1e-4')
    axes[1, 1].set_title('Elecricity Cost: 125 EUR/MWh\nField Tolerance: 1e-3')
    axes[1, 2].set_title('Elecricity Cost: 125 EUR/MWh\nField Tolerance: 1e-2')

    axes[1, 0].set_xlabel(r'Current Density in A/mm$^2$')
    axes[1, 1].set_xlabel(r'Current Density in A/mm$^2$')
    axes[1, 2].set_xlabel(r'Current Density in A/mm$^2$')
    axes[0, 0].set_ylabel(r'Cost in Million EUR')
    axes[1, 0].set_ylabel(r'Cost in Million EUR')

    # Add legend
    axes[0, 2].legend(loc='upper right')
    
    # remove x-axis labels from the top row
    for ax in axes[0]:
        ax.set_xticklabels([])
    
    # remove y-axis labels from the right column
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    for ax in axes[:, 2]:
        ax.set_yticklabels([])

    # fig.suptitle(r'\textbf{Cost estimates based on pure Analytical Modelling}')
    fig.suptitle(r'Cost estimates based on pure Analytical Modelling for Dipoles with $B_0$={}T, $aper_x$={}m, $aper_y$={}m and $L$={}m'.format(B_design, aper_x, aper_y, length), fontsize=16)

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.savefig('magnetcosts_analytical.png')
    plt.savefig('magnetcosts_analytical.pdf')
    plt.show()

if __name__ == '__main__':
    main()
