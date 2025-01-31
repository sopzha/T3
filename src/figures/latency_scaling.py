import json

from matplotlib import pyplot as plt

from src.figures.infra import setup_matplotlib_latex_font, get_figure_path, get_hex_color, get_figure_format


def latency_scaling_figure():
    setup_matplotlib_latex_font()

    with open("dp/latencyScalingCompiled.json", "r") as file:
        data1 = json.load(file)
    with open("dp/latencyScalingInterpretedST.json", "r") as file:
        data2 = json.load(file)
    with open("dp/latencyScalingInterpretedMT.json", "r") as file:
        data3 = json.load(file)

    x = [float(k) for k in data1.keys()]
    y1 = list(data1.values())
    y2 = list(data2.values())
    y3 = list(data3.values())

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    # mark avg query size
    avg_query_size = 3
    # print(f"avg query has {avg_query_size} pipelines and {data[str(avg_query_size)]} latency")
    ax.scatter(avg_query_size, y1[avg_query_size - 1], color=get_hex_color("my_yellow"))
    ax.annotate(
        "Avg Query",
        (avg_query_size, y1[avg_query_size - 1]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
        # color="grey"
    )

    ax.plot(x, y1, color=get_hex_color("my_blue"))
    ax.plot(x, y2, color=get_hex_color("my_red"))
    ax.plot(x, y3, color=get_hex_color("my_green"))

    ax.text(1000, y1[-1], "Compiled ST", ha="right", va="bottom", size=11, color="black")
    ax.text(240, y2[240], "Interpreted ST", ha="right", va="bottom", size=11, color="black")
    ax.text(1000, y3[-1], "Interpreted MT", ha="right", va="bottom", size=11, color="black")

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    # naming
    label = ax.set_ylabel("Latency in ms")
    fig.supxlabel("Number of Predicted Pipelines", color=label._color, font=label._fontproperties, y=0.1)

    # plt.show()
    #  return
    fig.savefig(f"{get_figure_path()}/latency_scaling.{get_figure_format()}", bbox_inches="tight")


def main():
    latency_scaling_figure()


if __name__ == "__main__":
    main()
