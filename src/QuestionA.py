import numpy as np
import matplotlib.pyplot as plt

font_dict = {'family': 'sans-serif', 'color': 'darkblue', 'weight': 'bold', 'size': 20}


def main():
    # Generate 100000 random numbers between 0 and 1
    random_numbers = np.random.rand(100000)
    plt.figure(figsize=(10, 6))
    plt.hist(random_numbers, bins=30, density=True, alpha=0.7, color='#a8dadc', edgecolor='#457b9d')
    plt.title('Histogram of Random Numbers', fontdict=font_dict)
    plt.xlabel('Value', fontdict=font_dict)
    plt.ylabel('Frequency', fontdict=font_dict)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('outputs/question_a_random_numbers_histogram_soft.png')

    # Average of the random numbers
    average = np.mean(random_numbers)
    print(f"Average of random numbers: {average}")
    # Standard deviation of the random numbers
    std_dev = np.std(random_numbers)
    print(f"Standard deviation of random numbers: {std_dev}")

    # Relative error compared to the expected value of 0.5
    relative_error = 100 * abs(average - 0.5) / 0.5
    print(f"Relative error: {relative_error:.2f}%")

    return 0


if __name__ == "__main__":
    main()