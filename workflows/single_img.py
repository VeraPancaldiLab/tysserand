import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tysserand import tysserand as ty
import cv2

# where we save and load annotations
data_dir = Path("./new_data")

## Adjust
img = plt.imread(data_dir / "1307.png")



# import random 
from random import sample

nodes = pd.read_csv(data_dir / '1307.csv', usecols=['x','y', 'class'])

nodes = nodes.sample(n = 1000)

coords = nodes.loc[:,['x','y']].values

class_colors = ['#F85446', '#813FE0', '#52DEF7', '#62E03F', '#FFCE36', '#FA6C17',
               '#D805E3', '#0155FA', '#0BE35B', '#FFF117', '#DDFF7D', '#78E3C8']
classes = list(nodes['class'].unique())
print(classes)
dico_col = {classes[0]:class_colors[0],
            classes[1]:class_colors[1],
            classes[2]:class_colors[2],
#            classes[3]:class_colors[3],
      #      classes[4]:class_colors[4],
           }
colors = []
for cl in nodes['class']:
    colors.append(dico_col[cl])
    

    
    
min_x = min(coords[:, 0])
min_y = min(coords[:, 1])

max_x = max(coords[:, 0])
max_y = max(coords[:, 1])

print( (max_x - min_x), (max_y - min_y)  )
print( (max_x - min_x), (max_y - min_y)  )

print(min_x, min_y)



coords2 = coords


coords2[:, 0] = coords2[:, 0] - 5550

coords2[:, 1] = coords2[:, 1] - 18470


img2 = cv2.resize(img, (19287 , 17222)) 
fig, ax = ty.showim(img2, figsize=(500, 500))




#pairs = ty.build_delaunay(coords2)
#distances = ty.distance_neighbors(coords2, pairs)
#col_nodes = colors
#ax.scatter(coords2[:,0], coords2[:,1], c=col_nodes,  zorder=10)
#plt.show()


##

#NAPARI
"""

import napari
viewer = napari.Viewer()
ty.visualize(viewer, img, colormaps='rgb')

 
annotations = ty.make_annotation_dict(
    coords, pairs=pairs,
    nodes_class=nodes['marker'],
    nodes_class_color_mapper=dico_col,
)
ty.add_annotations(viewer, annotations)

"""

