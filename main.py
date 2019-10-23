import sys

import tvm
import numpy as np

from capsule import get_workload

n = 24
i_h = 14
i_w = 14
o_h = 6
o_w = 6
i_size = 32
o_size = 32
kernel_size = 3
pose_size = 4
target = "cuda"

capsule, capsule_schedule = get_workload(target)

inputs = ["i_poses", "i_activations", "w", "beta_v", "beta_a"]
outputs = ["o_poses", "o_activations"]

shape = {}
shape["i_poses"] = (n, i_h, i_w, i_size, pose_size, pose_size)
shape["i_activations"] = (n, i_h, i_w, i_size)
shape["w"] = (kernel_size, kernel_size, i_size, o_size, pose_size, pose_size)
shape["beta_v"] = (o_size,)
shape["beta_a"] = (o_size,)
shape["o_poses"] = (n, o_h, o_w, o_size, pose_size, pose_size)
shape["o_activations"] = (n, o_h, o_w, o_size)

var = {}
for key in inputs:
    var[key] = tvm.placeholder(shape[key], dtype="float32", name=key)
var["o_poses"], var["o_activations"] = capsule(
        *map(lambda key: var[key], inputs),
        n, i_h, i_w, i_size, o_size, kernel_size, strides=2, iterations=3, pose_size=pose_size)
s = capsule_schedule(var["o_poses"], var["o_activations"], iterations=3)

varList = list(map(lambda key: var[key], inputs + outputs))
if "--print-lower" in sys.argv[1:]:
    print(tvm.lower(s, varList, simple_mode=True, name="run"))
compute = tvm.build(s, list(map(lambda key: var[key], inputs + outputs)), target, name="run")

ctx = tvm.context(target, 0)
data = {}
for key in inputs:
    data[key] = tvm.nd.array(np.random.uniform(size=shape[key]).astype("float32"), ctx)
for key in outputs:
    data[key] = tvm.nd.array(np.zeros(shape[key], dtype="float32"), ctx)

if "--print-source" in sys.argv[1:]:
    if target == "cuda" or target == "rocm" or target.startswith('opencl'):
        dev_module = compute.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(compute.get_source())

print()
print("Running...")
timer = compute.time_evaluator("run", ctx, number=1, repeat=5)
prof_res = np.array(timer(*map(lambda key: data[key], inputs + outputs)).results) * 1e3
print("Time cost is: ", np.mean(prof_res), "ms", " stddev = ", np.std(prof_res), "ms")

