#%%
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import trimesh

from tensorflow_graphics.nn.layer import graph_convolution as graph_conv
from tensorflow_graphics.notebooks import mesh_segmentation_dataio as dataio
from tensorflow_graphics.notebooks import mesh_viewer

path_to_model_zip = tf.keras.utils.get_file(
    'model.zip',
    origin='https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/model.zip',
    extract=True)

path_to_data_zip = tf.keras.utils.get_file(
    'data.zip',
    origin='https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/data.zip',
    extract=True)

local_model_dir = os.path.join(os.path.dirname(path_to_model_zip), 'model')
test_data_files = [
    os.path.join(
        os.path.dirname(path_to_data_zip),
        'brenda2.tfrecords')
]

test_io_params = {
    'is_training': False,
    'sloppy': False,
    'shuffle': True,
}
test_tfrecords = test_data_files

input_graph = tf.Graph()
with input_graph.as_default():
  mesh_load_op = dataio.create_input_from_dataset(
      dataio.create_dataset_from_tfrecords, test_tfrecords, test_io_params)
  with tf.Session() as sess:
    test_mesh_data, test_labels = sess.run(mesh_load_op)

input_mesh_data = {
    'vertices': test_mesh_data['vertices'][0, ...],
    'faces': test_mesh_data['triangles'][0, ...],
    'vertex_colors': mesh_viewer.SEGMENTATION_COLORMAP[test_labels[0, ...]],
}

mesh = trimesh.Trimesh(vertices=test_mesh_data['vertices'][0, ...],
                       faces=test_mesh_data['triangles'][0, ...],
                       vertex_colors= mesh_viewer.SEGMENTATION_COLORMAP[test_labels[0, ...]])
mesh.show()


# %%
