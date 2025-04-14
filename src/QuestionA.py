import numpy as np 
import matplotlib.pyplot as plt
#Show that the random number generator is reliable. 
# Does it yield to uniformly distributed between(0,1) ?

font_dict = {'family': 'serif', 'color':  'darkblue', 'weight': 'normal', 'size': 16,}


def main(): 
    # Generate 1000 random numbers between 0 and 1
    random_numbers = np.random.rand(100000)
    plt.figure(figsize=(10, 6))
    plt.hist(random_numbers, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of Random Numbers', fontdict=font_dict)
    plt.xlabel('Value', fontdict=font_dict)
    plt.ylabel('Frequency', fontdict=font_dict)
    plt.grid(True)
    plt.savefig('random_numbers_histogram.png')

    # Average of the random numbers
    average = np.mean(random_numbers)
    print(f"Average of random numbers: {average}")
    # Standard deviation of the random numbers
    std_dev = np.std(random_numbers)
    print(f"Standard deviation of random numbers: {std_dev}")

    # Relative error compared to the expected value of 0.5
    relative_error = 100*abs(average - 0.5) / 0.5
    print(f"Relative error: {relative_error}")

    return 0 

if __name__ == "__main__":
    main()
