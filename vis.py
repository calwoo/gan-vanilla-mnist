"""
MNIST visualizer for GANs.
"""

import os
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
# for making gif
import glob
import moviepy.editor as mpy 

def plot_mnist(samples):
    fig = plt.figure(figsize=(5,5))
    grid = gridspec.GridSpec(5,5)
    grid.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(grid[i])
        plt.axis("off")
        plt.imshow(sample.reshape(28,28))
    return fig

def take_snapshot(samples, marker=0):
    fig = plot_mnist(samples)
    
    # check if there is folder
    if not os.path.exists("./output"):
        os.mkdir("./output")
    
    # save image
    plt.savefig("./output/{}.png".format(str(marker).zfill(3)), bbox_inches="tight")
    plt.close(fig)

def build_gif(gif_name="output"):
    fps = 12
    # get all png files in directory
    file_list = glob.glob("./output/*.png")
    print(file_list)
    list.sort(file_list, key=lambda x: int(x.split(".")[1].split("/")[2]))
    clip = mpy.ImageSequenceClip(file_list, fps=fps)
    clip.write_gif("{}.gif".format(gif_name), fps=fps)