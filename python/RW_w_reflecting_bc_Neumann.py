import sys
import time
import numpy as np
import random 

import matplotlib.pyplot as plt 

# this is where the simulation will be hapenning
class Environment():

    def __init__(self, L, T_MAX):  
        self.lattice_size = L
        self.simulation_time = T_MAX
        self.walker_position = random.randint(0, L-1) # pick random lattice site
        self.flip_coin = random.randint(0,1)
        # record the data
        self.record_pdf = np.zeros(self.lattice_size)
        self.record_current = np.zeros(self.lattice_size)

    def run(self):
        for t_step in range(self.simulation_time):
            delta_x = 0
            self.flip_coin = random.randint(0,1)
            if self.flip_coin == 0: # jump to the right
                delta_x = 1
            else:                   # jump to the left
                delta_x = -1
     
            # impose reflective boundary conditions
            if self.walker_position + delta_x > self.lattice_size - 1 or self.walker_position + delta_x < 0:
                delta_x = -delta_x
            
            self.record_current[self.walker_position] += delta_x/(self.simulation_time)
            self.walker_position = self.walker_position + delta_x
            self.record_pdf[self.walker_position] += 1.0/(self.simulation_time)

        return  self.record_pdf, self.record_current


# our main
if __name__ == '__main__':
    start_time = time.time() # time your simulation
    L = 20           # lattice size
    T_MAX = 10000000    # total number of steps
    pdf = []
    current = []
    
    # run simulation
    simulation = Environment(L, T_MAX)
    pdf, current = simulation.run()
    print("Python execution time: %s seconds " % (time.time() - start_time))

    # record text output file
    shift = int(L/2)
    x_axis = np.arrange(0,L)
    x_axis_shifted = np.arange(-shift,shift)
    pdf_and_current_filename = "RW_Refl_output/PDF_Curret_L" + str(L) + "_TMAX" + str(T_MAX) + ".txt"
    np.savetxt(pdf_and_current_filename, np.c_[x_axis,pdf,current])

    # plot pdf 
    pdf_filename = "RW_Refl_output/PDF_L" + str(L) + "_TMAX" + str(T_MAX) + ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("X", fontsize=24)
    plt.ylabel("PDF", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    plt.plot(pdf, linestyle='--', marker='o', color='k')
    plt.tight_layout()
    plt.savefig(pdf_filename, format="png", dpi=600)
    #plt.show()
    
    # plot pdf shifted
    pdf_shifted_filename = "RW_Refl_output/PDF_shifted_L" + str(L) + "_TMAX" + str(T_MAX) + ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("X", fontsize=24)
    plt.ylabel("PDF", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    pdf = np.roll(pdf,shift) 
    plt.plot(x_axis_shifted, pdf, linestyle='--', marker='o', color='k')
    plt.tight_layout()
    plt.savefig(pdf_shifted_filename, format="png", dpi=600)
    #plt.show()
    
    # plot current shifted
    current_filename = "RW_Refl_output/Current_shifted_L" + str(L) + "_TMAX" + str(T_MAX) + ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("X", fontsize=24)
    plt.ylabel("Current", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    current = np.roll(current,shift) 
    plt.plot(x_axis_shifted,current, linestyle='--', marker='o', color='k')
    plt.tight_layout()
    plt.savefig(current_filename, format="png", dpi=600)
    #plt.show()