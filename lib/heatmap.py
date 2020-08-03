import math
import numpy as np

class HeatmapGenerator:
    def __init__(self, heatmap_size, sigma=1):
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    @staticmethod
    def Gaussian2D(x, y, center, sigma):
        exponent = (x-center[1])**2 + (y-center[0])**2
        if exponent > (3*sigma)**2:
            return 0
        exponent = -exponent / 2 / sigma**2
        return math.exp(exponent)# / 2 / math.pi / sigma**2

    def generate_target(self, pts):
        """
            return: a heatmap image.

        """
        heat_map = np.zeros((len(pts), self.heatmap_size, self.heatmap_size))

        for pt, heat in zip(pts,heat_map):
            for x in range(heat.shape[0]):
                for y in range(heat.shape[1]):
                    real_pt = pt*self.heatmap_size
                    heat[x][y] = HeatmapGenerator.Gaussian2D(x, y, real_pt, self.sigma)
        return heat_map

if __name__ == '__main__':
    hg = HeatmapGenerator(256, 16)
    pts = [[0.2,0.3],[0.57,0.5],[-0.05,0.5]]
    h = hg.generate_target(np.array(pts))
    from cv2 import cv2
    h=h.transpose((1,2,0))*255
    cv2.imwrite("heatmap.jpg", h)