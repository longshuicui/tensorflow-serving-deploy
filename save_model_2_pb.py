from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants, signature_def_utils, tag_constants, utils
from tensorflow.python.util import compat
import tensorflow as tf
import os

def save_model(sess,model,save_path,model_version):
    #模型签名，模型输入和输出的签名
    model_signature = signature_def_utils.build_signature_def(
        inputs={
            "inputs": utils.build_tensor_info(model.x),
            "mask": utils.build_tensor_info(model.mask)},
        outputs={
            "outputs": utils.build_tensor_info(model.y)},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    export_path = os.path.join(compat.as_bytes(save_path), compat.as_bytes(str(model_version)))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         clear_devices=True,
                                         signature_def_map={'bilstm_crf':model_signature},
                                         legacy_init_op=legacy_init_op)
    builder.save()
