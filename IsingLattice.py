import numpy as np
import math

class IsingLattice:

    E = 0.0
    E2 = 0.0
    M = 0.0
    M2 = 0.0

    n_cycles = 0
    
    n_cycles_to_ignore = 10000

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.lattice = np.random.choice([-1,1], size=(n_rows, n_cols))

    def energy(self):
        "Return the total energy of the current lattice configuration."
        "Loops through all positions in lattice, calculates interactions with neighbours assuming J=1"
        "Adds interactions of all positions to energy variable and returns it at the end"
        energy = 0.0
        J = 1
        #since we are already multiplying by 0.5 in the energy equation, we can remove a bunch of calculations if we just shift only down and to the right.
        #in the previous code we were also shifting up and to the left, but we can just choose not to use the 0.5 factor as this method will still take into
        # consideration all the interactions between the spin states.
        spin_right = np.roll(self.lattice,1,1)
        spin_down = np.roll(self.lattice,1,0)
        #now we can simply np.multiply these shifted lattices by the old lattice to get the new lattice configs. Bear in mind that this will need another dimension 
        #if we switch to a 3D lattice.
        right_interactions = np.multiply(spin_right, self.lattice)
        down_interactions = np.multiply(spin_down, self.lattice)
        #notice that now we have removed the need for looping between each state. this is a much quicker algorithm. Also, we will no longer need to multiply by 0.5.
        energy = (-1)*J*(np.sum(right_interactions)+np.sum(down_interactions))
        return energy

    def magnetisation(self):
        "Return the total magnetisation of the current lattice configuration. This is simply the sum of all spins in the system"
        magnetisation = np.sum(self.lattice)
        return magnetisation


    def montecarlostep(self, T):
        # complete this function so that it performs a single Monte Carlo step
        energy_0 = self.energy()
        #the following two lines will select the coordinates of the random spin for you
        random_i = np.random.choice(range(0, self.n_rows))
        random_j = np.random.choice(range(0, self.n_cols))
        # flip the spin at a random coordinate in the lattice.
        self.lattice[random_i, random_j] = self.lattice[random_i, random_j]*(-1)
        #calculate new energy & energy difference
        energy1 = self.energy()
        delta_e = energy1 - energy_0
        #deciding if to accept the new config or not
        if delta_e>0:
            #the following line will choose a random number in the range[0,1) for you
            random_number = np.random.random()
            # we are using reduced energy quantities, so will not be using boltzmann constant in the boltzmann probability
            if random_number > np.exp(-delta_e/T):
                #new config is rejected and we need to restore the prev lattice 
                self.lattice[random_i, random_j] = self.lattice[random_i, random_j]*(-1)
                energy1 = energy_0
        #update the sum of energies & energies squared & magnetisation and magnetisation squared, only if #cycles is greater than cycles to ignore
        #we don't want it to update it if the target cycles hasn't been reached yet.
        mag = self.magnetisation()
        #update the number of cycles
        self.n_cycles += 1
        if self.n_cycles > self.n_cycles_to_ignore:
            self.E += energy1
            self.E2 += energy1**2
            self.M += mag
            self.M2 += mag**2
        #return the new energy and magnetisation                
        return energy1, mag

 

    def statistics(self):
        # complete this function so that it calculates the correct values for the averages of E, E*E (E2), M, M*M (M2), and returns them with Nsteps
        # the monte carlo algorithm replaces the partition function here, so we divide by the number of MC steps rather than possible configurations as we would with partition.
        ave_E = self.E/(self.n_cycles - self.n_cycles_to_ignore)
        ave_E2 = self.E2/(self.n_cycles - self.n_cycles_to_ignore)
        ave_M = self.M/(self.n_cycles - self.n_cycles_to_ignore)
        ave_M2 = self.M2/(self.n_cycles - self.n_cycles_to_ignore)
        return ave_E, ave_E2, ave_M, ave_M2, self.n_cycles
