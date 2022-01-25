import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set()


def main():
    fig = plt.figure()

    colors = ['r', 'b', 'g']
    with open('points.yml', 'r') as f:
        metric_points = yaml.safe_load(f)

    color_index = 0
    for metric, points in metric_points.items():
        plt.plot(points, color=colors[color_index], label=f"{metric} fitness value")
        color_index += 1

    plt.xlabel("Generation Number")
    plt.ylabel("Fitness")
    plt.legend()
    fig.savefig('Fitness.png', dpi=720)


if __name__ == "__main__":
    main()
