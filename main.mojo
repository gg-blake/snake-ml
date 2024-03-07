from python import Python
from population import Population

fn main() raises:
    print("Initializing population...")
    var population = Population[100]()
    print("Population completed successfully.\nStarting simulation...")
    var timeout = 100
    while population.update_habitat() and timeout > 0:
        print("Simulating population...", "Time Until Time Out:", timeout)
        timeout -= 1

    print("Simulation complete.\nGenerating next habitat...")
    population.generate_next_habitat(survival_rate=0.5)
    print("New habitat generated!")

    