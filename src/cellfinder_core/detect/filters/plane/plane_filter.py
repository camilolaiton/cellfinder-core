from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cellfinder_core.detect.filters.plane.classical_filter import enhance_peaks
#from cellfinder_core.detect.filters.plane.tile_walker import TileWalker


@dataclass
class TileProcessor:
    clipping_value: float
    threshold_value: float
    soma_diameter: float
    log_sigma_size: float
    n_sds_above_mean_thresh: float
    process_by: str

    def get_tile_mask(
        self, plane: np.ndarray, stats: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warning: this modifies ``plane`` in place.
        """
        laplace_gaussian_sigma = self.log_sigma_size * self.soma_diameter
        plane = plane.T
        np.clip(plane, 0, self.clipping_value, out=plane)

        #walker = TileWalker(plane, self.soma_diameter, self.process_by)

        #walker.walk_out_of_brain_only()

        thresholded_img = enhance_peaks(
            plane.copt(), #walker.thresholded_img,
            self.clipping_value,
            gaussian_sigma=laplace_gaussian_sigma,
            plane_max=stats[1]
        )

        # threshold
        if isinstance(stats, type(None)):
            avg = thresholded_img.ravel().mean()
            sd = thresholded_img.ravel().std()
        else:
            avg = stats[2]
            sd = stats[3]

        plane[
            thresholded_img > avg + self.n_sds_above_mean_thresh * sd
        ] = self.threshold_value
        return plane, np.ones((plane.shape), dtype="uint8") #walker.good_tiles_mask.astype(np.uint8)
