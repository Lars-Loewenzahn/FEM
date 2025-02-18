# Import necessary libraries for numerical operations, file handling, and visualization
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import openmm as mm
import openmm.app as app

# Define constants for the simulation
chain_length = 10  # Number of segments in the chain
framerate = 300  # Frames per second for the animation
trajectory_step = 1  # Simulation steps per frame
# Time rescaling factor for the simulation
# Epsilon is a parameter for the Lennard-Jones potential
# Data folder to store simulation results
time_rescaling = 1/15
epsilon = 1
data_folder = "data"

# name = "tapered"
# radii = np.linspace(0.005, 0.005/3, chain_length+1)
# masses = 4 / 3 * np.pi * np.power(radii, 3)
# masses /= np.max(masses) / 0.001
# print("Masses: ", masses)
# framerate = 30
# trajectory_step = 10
# n_frames = 500

print("Checkpoint 1")

# Linear chain configuration
name = "linear"
radii = np.ones(chain_length+1) * 0.005  # Radii of the chain segments in meters
masses = np.ones(chain_length+1) * 0.001  # Masses of the chain segments in kilograms
# Define the number of frames for the simulation
n_frames = 500
# Set the framerate and trajectory step for the linear chain
framerate = 100
trajectory_step = 50
# Define the time array for the simulation
t = np.linspace(0, n_frames/framerate, n_frames)

print("Checkpoint 2")

masses[0] = 0     # Fix the first segment to prevent movement
# Create a topology for the chain
# Add a chain and residue to the topology
topology = app.Topology()
chain = topology.addChain()
residue = topology.addResidue("res", chain)

for i, mass in enumerate(masses):
    # Define a custom element for each segment
    element = mm.app.Element(number=1000,
                             name=f"Whipium{i}",
                             symbol=f"Wh{i}",
                             mass=mass / 1.660539e-9)  # Convert kg to 10e18 kg/mol
    topology.addAtom(name="Whipium", element=element, residue=residue)

print("Checkpoint 3")

# Create a system and integrator for the simulation
system = mm.System()
# Define an integrator with a specific time step
integrator = mm.VerletIntegrator(1000 / framerate / trajectory_step * time_rescaling)

# Add gravity as a custom external force
# Define the gravity force equation and add it to the system
gravity = mm.CustomExternalForce("m*z*9.81e-6")  # Acceleration in m/ms²
m_index = gravity.addPerParticleParameter("m")
for i, atom in enumerate(topology.atoms()):
    if i > 0:
        gravity.addParticle(i, [atom.element.mass])
system.addForce(gravity)

print("Checkpoint 4")

# Add harmonic bonds between consecutive segments
bonds = mm.HarmonicBondForce()
for i in range(chain_length - 1):
    bonds.addBond(i, i + 1, length=0.01, k=1e5)
system.addForce(bonds)

# Add a repulsion force between non-consecutive segments
repulsion = mm.CustomNonbondedForce(f"4*{epsilon}*((0.5*(sigma1+sigma2)/r)^12)")
repulsion.addPerParticleParameter("sigma")
for i, atom in enumerate(topology.atoms()):
    repulsion.addParticle([radii[i]])
    if i > 0:
        repulsion.addExclusion(i-1, i)
system.addForce(repulsion)

print("Checkpoint 5")

# Add particles to the system based on the defined masses
for i, atom in enumerate(topology.atoms()):
    system.addParticle(mass=atom.element.mass)

# Define initial positions for the chain segments
marble_positions = [[0, 0, -i*0.01] for i in range(chain_length + 1)]

# Create a simulation object with the defined topology, system, and integrator
sim = app.Simulation(topology, system, integrator)
# Set the initial positions of the segments in the simulation
sim.context.setPositions(marble_positions)

print("Checkpoint 6")

# Initialize lists to store simulation data
velocities = []
trajectory = []
pot_energy = []
kinetic_energy = []
forces = []
# Run the simulation for the defined number of frames
for i in range(n_frames):
    sim.step(trajectory_step)  # Perform a simulation step
    # Get the state of the system including positions, velocities, energies, and forces
    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
    # Append the data to the respective lists
    trajectory.append(state.getPositions(asNumpy=True))
    velocities.append(state.getVelocities(asNumpy=True))
    pot_energy.append(state.getPotentialEnergy()._value)
    kinetic_energy.append(state.getKineticEnergy()._value)
    forces.append(state.getForces(asNumpy=True))

print("Checkpoint 7")

# Code to save simulation data to files (commented out)
"""
if not os.path.exists(f"{data_folder}/{name}/"):
    os.makedirs(f"{data_folder}/{name}/")

np.save(f"{data_folder}/{name}/traj.npy", trajectory)
np.save(f"{data_folder}/{name}/velocities.npy", velocities)
np.save(f"{data_folder}/{name}/pot_energy.npy", pot_energy)
np.save(f"{data_folder}/{name}/kinetic_energy.npy", kinetic_energy)
np.save(f"{data_folder}/{name}/forces.npy", forces)
np.save(f"{data_folder}/{name}/masses.npy", masses)
np.save(f"{data_folder}/{name}/radii.npy", radii)

"""

print("Checkpoint 8")

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.1, 0.1)
ln, = plt.plot([], [], 'ro-', lw=2)  # Initialize the line plot for the chain

x_data, y_data = [], []  # Lists to store x and y coordinates of chain segments

print("Checkpoint 9")
# Print the first and last frame of the trajectory for debugging
print(trajectory[0])
print(trajectory[-1])

# Extract x and y coordinates from the trajectory data
for index, frame_data in enumerate(trajectory):
    chain_x, chain_y = [], []
    for chainlink in frame_data:
        x = chainlink[0].value_in_unit(mm.unit.nanometer)  # Convert x-coordinate to nanometers
        y = chainlink[2].value_in_unit(mm.unit.nanometer)  # Convert y-coordinate to nanometers
        chain_x.append(x)
        chain_y.append(y)
    x_data.append(chain_x)
    y_data.append(chain_y)

# Initialize the animation
# Set the initial data for the line plot
def init():
    ln.set_data([], [])
    return ln,

# Update function for the animation
# Set the data for the current frame
def update(frame):
    ln.set_data(x_data[frame], y_data[frame])
    return ln,

# Create and save the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
ani.save('pen.gif',writer='pillow',fps=framerate)  # Save the animation as a GIF
print("All done!")