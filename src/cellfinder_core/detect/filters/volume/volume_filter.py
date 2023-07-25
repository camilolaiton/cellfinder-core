import os
import math
import logging
import threading

from typing import Callable, List, Sequence
#from multiprocessing.pool import AsyncResult

import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import tifffile
from collections import defaultdict
from imlib.cells.cells import Cell

from cellfinder_core.detect.filters.setup_filters import setup
from cellfinder_core.detect.filters.volume.structure_detection import (
    get_structure_centre_wrapper,
)
from cellfinder_core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)



# multithreading the volume filter: NL 12/21/22
def process_thread(detector, middle_plane):
    detector.process(middle_plane)

class VolumeFilter(object):
    def __init__(
        self,
        *,
        soma_diameter: float,
        soma_size_spread_factor: float = 1.4,
        setup_params: Sequence,
        planes_paths_range: Sequence,
        save_planes: bool = False,
        plane_directory: str = None,
        start_plane: int = 0,
        max_cluster_size: int = 5000,
        outlier_keep: bool = False,
        artifact_keep: bool = True,
        block: int,
        holdover: dict = None
    ):
        self.soma_diameter = soma_diameter
        self.soma_size_spread_factor = soma_size_spread_factor
        self.planes_paths_range = planes_paths_range
        self.z = start_plane
        self.save_planes = save_planes
        self.plane_directory = plane_directory
        self.max_cluster_size = max_cluster_size
        self.outlier_keep = outlier_keep

        self.artifact_keep = artifact_keep

        self.clipping_val = None
        self.threshold_value = None
        self.setup_params = setup_params
        
        # Needed to handle ROIs present at data splits: NL 12/07/22
        self.holdover = holdover
        
        
        if isinstance(self.holdover, type(None)):
            self.ball_filter, self.cell_detector  = setup(
                self.setup_params[0],
                self.setup_params[1],
                self.setup_params[2],
                self.setup_params[3],
                ball_overlap_fraction=self.setup_params[4],
                z_offset=self.setup_params[5],
            )
        elif block == 0:
            self.ball_filter, self.cell_detector  = setup(
                self.setup_params[0],
                self.setup_params[1],
                self.setup_params[2],
                self.setup_params[3],
                ball_overlap_fraction=self.setup_params[4],
                z_offset=self.setup_params[5],
            )
            self.holdover['cells'] = {}
        else:
            self.ball_filter = self.holdover['ball_filt']
            self.z = self.holdover['z']
            self.setup_params[5] = self.holdover['z'] - int(math.floor(self.setup_params[3] / 2)) - 1
            
            _, self.cell_detector  = setup(
                self.setup_params[0],
                self.setup_params[1],
                self.setup_params[2],
                self.setup_params[3],
                ball_overlap_fraction=self.setup_params[4],
                z_offset=self.setup_params[5],
            )
            
            previous_img, struct_id = self.holdover['previous_img'], self.holdover['next_id']
            self.cell_detector.set_previous_params(previous_img, struct_id, self.holdover['cells'])
        
    def process(
        self,
        async_results: List[list], #List[AsyncResult],
        callback: Callable[[int], None],
    ):
        
        progress_bar = tqdm(
            total=len(async_results), desc="Processing planes"
        )
        
        t = 0
        
        for plane_id, res in enumerate(async_results):
            plane, mask = res[0], res[1]#res.get()
            logging.debug(f"Plane {plane_id} received for 3D filtering")
            print(f"Plane {plane_id} received for 3D filtering")

            logging.debug(f"Adding plane {plane_id} for 3D filtering")
            self.ball_filter.append(plane, mask)
            
            if self.ball_filter.ready:

                logging.debug(f"Ball filtering plane {plane_id}")
                self.ball_filter.walk()

                middle_plane = self.ball_filter.get_middle_plane()
                
                if self.save_planes:
                    self.save_plane(middle_plane)
                
                # make sure previous thread is finished prior to creating new one
                if isinstance(t, int):
                    pass
                else:
                    t.join()
                    
                logging.debug(f"Detecting structures for plane {plane_id}")
                self.cell_detector.process(middle_plane)
                
                '''
                Removing threading since it is being run within a delayed 
                function and cannot do both
                # Start on separate thread so the next ballfilter.walk() instance can start: NL 12/21/22
                t = threading.Thread(target = process_thread, args = (self.cell_detector, middle_plane))
                t.start()
                '''
                
                logging.debug(f"Structures done for plane {plane_id}")
                logging.debug(
                    f"Skipping plane {plane_id} for 3D filter"
                    " (out of bounds)"
                )

            callback(self.z)
            self.z += 1
            progress_bar.update()
        
        # get parameters to holdover for next loop
        if isinstance(self.holdover, dict):
            self.holdover["ball_filt"] = self.ball_filter
            self.holdover["z"] =  self.z
            self.holdover["previous_img"] = self.cell_detector.get_previous_layer()
            self.holdover["next_id"] = self.cell_detector.get_next_structure_id()

        
        progress_bar.close()
        #t.join()
        logging.debug("3D filter done")
        

        cells = self.get_results()

        return cells, self.holdover

    def save_plane(self, plane):
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)      

    # might want to multiprocess later, but overall time  in function is small
    def get_results(self):
        logging.info("Splitting cell clusters and writing results")

        max_cell_volume = sphere_volume(
            self.soma_size_spread_factor * self.soma_diameter / 2
        )

        cells = []
        for (
            cell_id,
            cell_points,
        ) in self.cell_detector.get_coords_list().items():
            cell_volume = len(cell_points)
            
            #If ROI has instance on plane before a split, save for next chunk and skip rest of function: NL 12/07/22
            cell_loc = pd.DataFrame(cell_points)
            
            if isinstance(self.holdover, dict) and max(cell_loc['z']) == self.z - int(math.floor(self.setup_params[3] / 2)):
                self.holdover['cells'][cell_id] = cell_points
                continue 
                
            if cell_volume < max_cell_volume:
                cell_centre = get_structure_centre_wrapper(cell_points)
                cells.append(
                    Cell(
                        (cell_centre["x"], cell_centre["y"], cell_centre["z"]),
                        Cell.UNKNOWN,
                    )
                )
            else:
                if cell_volume < self.max_cluster_size:
                    try:
                        cell_centres = split_cells(
                            cell_points, outlier_keep=self.outlier_keep
                        )
                    except (ValueError, AssertionError) as err:
                        raise StructureSplitException(
                            f"Cell {cell_id}, error; {err}"
                        )
                    for cell_centre in cell_centres:
                        cells.append(
                            Cell(
                                (
                                    cell_centre["x"],
                                    cell_centre["y"],
                                    cell_centre["z"],
                                ),
                                Cell.UNKNOWN,
                            )
                        )
                else:
                    cell_centre = get_structure_centre_wrapper(cell_points)
                    cells.append(
                        Cell(
                            (
                                cell_centre["x"],
                                cell_centre["y"],
                                cell_centre["z"],
                            ),
                            Cell.ARTIFACT,
                        )
                    )
                    
        return cells

def sphere_volume(radius):
    return (4 / 3) * math.pi * radius**3
