import pylab as pl
import seaborn as sns


def mel(f):
    return 2595 * pl.log10(1 + f / 700)


def main():
    sns.set()

    y = [mel(x) for x in range(16000)]

    fig, ax = pl.subplots(1, 1, figsize=(8, 4))
    ax.plot(y, )
    ax.set_xlabel('hertz')
    ax.set_ylabel('mels')
    fig.show()


if __name__ == '__main__':
    main()
