#%% 
import os
import trimesh
import argparse
import numpy as np
import tensorflow as tf

#%% 
def parse_obj(file_path):
    vertices = []
    triangles = []
    with open(file_path) as fp:
        for line in fp:
            parsed = _parse_vertex_or_triangle(line)
            if parsed[0] == 'v':
                vertices.append(parsed[1])
            if parsed[0] == 'f':
                triangles.append(parsed[1])
    num_vertices = len(vertices)
    num_triangles = len(triangles)

    vertices = tf.constant(np.vstack(vertices))
    triangles = tf.constant(np.vstack(triangles))
    return num_vertices, num_triangles, vertices, triangles


def _parse_vertex_or_triangle(line):
    elem_type = None
    data = np.zeros(3)
    if not line or not line == '\n':  # check if line is empty
        separated = line.split()
        if separated[0] == 'v':
            elem_type = 'v'
            data[0] = float(separated[1])
            data[1] = float(separated[2])
            data[2] = float(separated[3])
        if separated[0] == 'f':
            elem_type = 'f'
            data[0] = int(separated[1].split('/')[0]) - 1
            data[1] = int(separated[2].split('/')[0]) - 1
            data[2] = int(separated[3].split('/')[0]) - 1
    return elem_type, data


def _list_string_feature(values):
    """Returns a bytes_list from a list of strings"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _tensor_feature(values, dtype):
    values = tf.dtypes.cast(values, dtype)
    serialised_values = tf.io.serialize_tensor(values)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialised_values.numpy()]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord_from_obj(obj_path, out_path):
    num_vertices, num_triangles, vertices, triangles = parse_obj(obj_path)
    labels = tf.ones(num_vertices, dtype=tf.int32)
    feature = {
                'num_vertices': _int64_feature(num_vertices),
                'num_triangles': _int64_feature(num_triangles),
                'vertices': _tensor_feature(vertices, tf.float32),
                'triangles': _tensor_feature(triangles, tf.int32),
                'labels': _tensor_feature(labels, tf.int32)
              }

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    serialized_proto = example_proto.SerializeToString()
    with tf.io.TFRecordWriter(out_path) as writer:
        writer.write(serialized_proto)


def batch_writing_tfrecord_from_obj(objs_folder, out_path, name_depth=1):
    # find all objs in a folder
    obj_files_path = []
    for dirpath, _, fnames in os.walk(objs_folder):
        for f in fnames:
            if f.endswith(".obj"):
                obj_files_path.append(os.path.join(dirpath, f))
    for obj_file_path in obj_files_path:
        tfrecord_file_name = obj_file_path.split('/')[-name_depth]
        if name_depth == 1:
            tfrecord_file_name = tfrecord_file_name.split('.')[0]
        tfrecord_file_path = out_path + tfrecord_file_name + '.tfrecord'
        write_tfrecord_from_obj(obj_file_path, tfrecord_file_path)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='brenda2.ply', help="Path to mesh to be converted")
args = parser.parse_args()

if __name__ == '__main__':
    file_name = args.name
    bla = trimesh.load_mesh(f'{file_name}.ply')
    bla.apply_scale(.001)
    bla.export(f'{file_name}.obj')
    write_tfrecord_from_obj(f'{file_name}.obj', f'{file_name}.tfrecords')
# %%
