import time
from matplotlib import pyplot as plt
import numpy as np


class LitboxDenoiserDisplay:
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.fig = fig
        self.axes = axes

    def show(self, input_image, output_image, target_image):
        # Ensure single-channel inputs (keep first channel if multiple) for the input plot
        if input_image.shape[1] != 1:
            input_plot = input_image[:, 0:1, :, :]
        else:
            input_plot = input_image

        # For output and target, keep full channels but ensure we display only first channel if required
        if output_image.shape[1] != 1:
            out_plot = output_image[:, 0:1, :, :]
        else:
            out_plot = output_image

        if target_image.shape[1] != 1:
            tgt_plot = target_image[:, 0:1, :, :]
        else:
            tgt_plot = target_image

        def to_numpy(img):
            arr = img[0].cpu().detach().numpy().transpose(1, 2, 0)  # H, W, C
            if arr.shape[2] == 1:  # grayscale -> squeeze to H, W
                arr = arr[:, :, 0]
            return arr

        a = to_numpy(input_plot)
        b = to_numpy(out_plot)
        c = to_numpy(tgt_plot)

        # Left: input image
        ax0 = self.axes[0]
        ax0.clear()
        # if a.ndim == 2:
        #     ax0.imshow(a, cmap='gray')
        # else:
        ax0.imshow(a)
        ax0.set_title("Input")
        ax0.axis('off')

        # Right: concatenated output + target (side-by-side)
        # Ensure same height
        if b.shape[0] != c.shape[0]:
            raise ValueError("Output and target must have the same height to concatenate")
        concat = np.concatenate([b, c], axis=1)

        ax1 = self.axes[1]
        ax1.clear()
        # if concat.ndim == 2:
        #     ax1.imshow(concat, cmap='gray')
        # else:
        ax1.imshow(concat)
        ax1.set_title("Output | Target")
        ax1.axis('off')

        # Re-draw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)  # Allow the plot to update

    def shutdown(self):
        plt.ioff()
        plt.show()  # Keep the final plot open
