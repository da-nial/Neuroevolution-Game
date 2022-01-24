import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set()


def main():
    colors = ['r', 'b', 'g']

    with open('points.yml', 'r') as f:
        metric_points = yaml.safe_load(f)

    color_index = 0
    for metric, points in metric_points.items():
        color_index += 1
        plt.plot(points, color=colors[color_index], label=f"{metric} fitness value")

    plt.xlabel("Generation Number")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
