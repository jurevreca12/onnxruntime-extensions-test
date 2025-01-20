import onnx
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path
import numpy as np

DOMAIN="ai.onnx.contrib"
#DOMAIN="qonnx.ops"

@onnx_op(op_type="CustomOpOne",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float])
def custom_one_op(x, y):
    return np.add(x, y)

def make_model():
    nodes = []
    nodes.append(
        helper.make_node(
            'CustomOpOne', 
            ['input_1', 'input_2'], 
            ['output'],
            domain=DOMAIN
        )
    )
    input_1 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [3,]
    )
    input_2 = helper.make_tensor_value_info(
        'input_2', onnx_proto.TensorProto.FLOAT, [3,]
    )
    output = helper.make_tensor_value_info(
        'output', onnx_proto.TensorProto.FLOAT, [3,]
    )
    graph = helper.make_graph(nodes, 'test0', [input_1, input_2], [output])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(DOMAIN, 1)], ir_version=7
    )
    return model

so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())
onnx_model = make_model()
sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
input_1 = np.array([1, 2, 3]).astype(np.float32)
input_2 = np.array([1, 1, 2]).astype(np.float32)
txout = sess.run(None, {'input_1': input_1, 'input_2': input_2})
print(txout)
