import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_palette("colorblind", 3)
sns.set_context("notebook")

parser = argparse.ArgumentParser()
parser.add_argument('--csv-path', type=str,
                    help='path to the low-frequency results in a csv format')
parser.add_argument('--out-dir', type=str,
                    help='path to save the plots')

def check_path(path):
    if path.endswith("/"):
        return path[:-1]
    else:
        return path

def plot(k, df, out_dir):

    x = [i*14 for i in range(1, 12)]
    y_style_trans = df[(df.experiment=='style_transfer')&(df.kFold==k)].auroc.tolist()
    y_stain_aug = df[(df.experiment=='stain_augmentation')&(df.kFold==k)].auroc.tolist()
    y_stain_norm = df[(df.experiment=='stain_normalization')&(df.kFold==k)].auroc.tolist()

    plt.plot(x, y_style_trans, label='style transfer (STRAP)')
    plt.plot(x, y_stain_aug, label='stain augmentation (SA)')
    plt.plot(x, y_stain_norm, label='stain normalization (SN)')

    plt.legend(loc='lower right')
    plt.xlabel('radius')
    plt.ylabel('AUROC')
    plt.xticks(x)
    plt.title(f'CV fold {k}')
    plt.savefig(out_dir + f'/plot_fold_{k}.pdf')
    print(f'successfully saved a plot for cv fold {k}')

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = check_path(args.out_dir)

    df = pd.read_csv(args.csv_path)

    for i in range(4):
        k = i+1
        plot(k, df, out_dir)