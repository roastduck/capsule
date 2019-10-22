import tvm
import topi

epsilon = 1e-5

def square(x):
    return x * x

capsule_impls = {}

''' Register a Capsule implementation for a specific target '''
def register_capsule_impl(target):
    def f(cls):
        capsule_impls[target] = cls
    return f

def get_workload(target):
    obj = capsule_impls[target]()
    algo = lambda *args, **kvs: obj.capsule(*args, **kvs)
    schedule = lambda *args, **kvs: obj.capsule_schedule(*args, **kvs)
    return algo, schedule

class Capsule:
    def __init__(self):
        self.tensors = {}

    def compute(self, shape, function, name=None):
        if name is None:
            return tvm.compute(shape, function)
        else:
            assert name not in self.tensors, "Name %s already exists" % name
            t = tvm.compute(shape, function, name="_I_".join(name.split("/")))
            self.tensors[name] = t
            return t

    def find(self, name):
        return self.tensors[name]

    ''' @param rr: shape(n, o_h, o_w, kernel_size, kernel_size, i_size, o_size)
        @param votes: shape(n, o_h, o_w, kernel_size, kernel_size, i_size, o_size, pose_size, pose_size)
        @param i_activations: shape(n, o_h, o_w, kernel_size, kernel_size, i_size)
        @param beta_v: shape(o_size)
        @param beta_a: shape(o_size) '''
    def m_step(self, it, rr, votes, i_activations, beta_v, beta_a, inverse_temperature, n, o_h, o_w, i_size, o_size, kernel_size, pose_size):
        rr_prime = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
                lambda nn, hh, ww, kx, ky, ci, co: rr[nn, hh, ww, kx, ky, ci, co] * i_activations[nn, hh, ww, kx, ky, ci],
                name="m_step/it_%d/rr_prime" % it)

        kx_axis, ky_axis, ci_axis = map(tvm.reduce_axis, [(0, kernel_size), (0, kernel_size), (0, i_size)])
        rr_prime_sum = self.compute((n, o_h, o_w, o_size),
                lambda nn, hh, ww, co: tvm.sum(
                    rr_prime[nn, hh, ww, kx_axis, ky_axis, ci_axis, co],
                    axis=[kx_axis, ky_axis, ci_axis]),
                name="m_step/it_%d/rr_prime_sum" % it)

        kx_axis, ky_axis, ci_axis = map(tvm.reduce_axis, [(0, kernel_size), (0, kernel_size), (0, i_size)])
        o_mean = self.compute((n, o_h, o_w, o_size, pose_size, pose_size),
                lambda nn, hh, ww, co, px, py: tvm.sum(
                    rr_prime[nn, hh, ww, kx_axis, ky_axis, ci_axis, co] * votes[nn, hh, ww, kx_axis, ky_axis, ci_axis, co, px, py],
                    axis=[kx_axis, ky_axis, ci_axis]),
                name="m_step/it_%d/o_mean_1" % it)
        o_mean = self.compute((n, o_h, o_w, o_size, pose_size, pose_size),
                lambda nn, hh, ww, co, px, py: o_mean[nn, hh, ww, co, px, py] / rr_prime_sum[nn, hh, ww, co],
                name="m_step/it_%d/o_mean_2" % it)

        kx_axis, ky_axis, ci_axis = map(tvm.reduce_axis, [(0, kernel_size), (0, kernel_size), (0, i_size)])
        o_stdv = self.compute((n, o_h, o_w, o_size, pose_size, pose_size),
                lambda nn, hh, ww, co, px, py: tvm.sum(
                    rr_prime[nn, hh, ww, kx_axis, ky_axis, ci_axis, co] * square(
                        votes[nn, hh, ww, kx_axis, ky_axis, ci_axis, co, px, py] - o_mean[nn, hh, ww, co, px, py]),
                    axis=[kx_axis, ky_axis, ci_axis]),
                name="m_step/it_%d/o_stdv_1" % it)
        o_stdv = self.compute((n, o_h, o_w, o_size, pose_size, pose_size),
                lambda nn, hh, ww, co, px, py: tvm.sqrt(o_stdv[nn, hh, ww, co, px, py]) / rr_prime_sum[nn, hh, ww, co],
                name="m_step/it_%d/o_stdv_2" % it)

        o_cost_h = self.compute((n, o_h, o_w, o_size, pose_size, pose_size),
                lambda nn, hh, ww, co, px, py: (beta_v[co] + tvm.log(o_stdv[nn, hh, ww, co, px, py] + epsilon)) * rr_prime_sum[nn, hh, ww, co],
                name="m_step/it_%d/o_cost_h" % it)

        px_axis, py_axis, = map(tvm.reduce_axis, [(0, pose_size), (0, pose_size)])
        o_cost = self.compute((n, o_h, o_w, o_size),
                lambda nn, hh, ww, co: tvm.sum(
                    o_cost_h[nn, hh, ww, co, px_axis, py_axis],
                    axis=[px_axis, py_axis]),
                name="m_step/it_%d/o_cost" % it)

        co_axis = tvm.reduce_axis((0, o_size))
        o_cost_mean = self.compute((n, o_h, o_w),
                lambda nn, hh, ww: tvm.sum(
                    o_cost[nn, hh, ww, co_axis],
                    axis=[co_axis]),
                name="m_step/it_%d/o_cost_mean_1" % it)
        o_cost_mean = self.compute((n, o_h, o_w),
                lambda nn, hh, ww: o_cost_mean[nn, hh, ww] / o_size,
                name="m_step/it_%d/o_cost_mean_2" % it)

        co_axis = tvm.reduce_axis((0, o_size))
        o_cost_stdv = self.compute((n, o_h, o_w),
                lambda nn, hh, ww: tvm.sum(
                    square(o_cost[nn, hh, ww, co_axis] - o_cost_mean[nn, hh, ww]),
                    axis=[co_axis]),
                name="m_step/it_%d/o_cost_stdv_1" % it)
        o_cost_stdv = self.compute((n, o_h, o_w),
                lambda nn, hh, ww: tvm.sqrt(o_cost_stdv[nn, hh, ww] / o_size),
                name="m_step/it_%d/o_cost_stdv_2" % it)
        o_activations = self.compute((n, o_h, o_w, o_size),
                lambda nn, hh, ww, co: tvm.sigmoid(
                    (beta_a[co] + (o_cost_mean[nn, hh, ww] - o_cost[nn, hh, ww, co]) / (o_cost_stdv[nn, hh, ww] + epsilon)) * inverse_temperature),
                name="m_step/it_%d/o_activations" % it)

        return o_mean, o_stdv, o_activations

    ''' @param o_mean: shape(n, o_h, o_w, o_size, pose_size, pose_size)
        @param o_stdv: shape(n, o_h, o_w, o_size, pose_size, pose_size)
        @param o_activations: shape(n, o_h, o_w, o_size)
        @param votes: shape(n, o_h, o_w, kernel_size, kernel_size, i_size, o_size, pose_size, pose_size) '''
    def e_step(self, it, o_mean, o_stdv, o_activations, votes, n, o_h, o_w, i_size, o_size, kernel_size, pose_size):
        px_axis, py_axis, = map(tvm.reduce_axis, [(0, pose_size), (0, pose_size)])
        o_p = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
                lambda nn, hh, ww, kx, ky, ci, co: tvm.sum(
                    -square(votes[nn, hh, ww, kx, ky, ci, co, px_axis, py_axis] - o_mean[nn, hh, ww, co, px_axis, py_axis]) /
                        (2 * square(o_stdv[nn, hh, ww, co, px_axis, py_axis]))
                    -tvm.log(o_stdv[nn, hh, ww, co, px_axis, py_axis] + epsilon),
                    axis=[px_axis, py_axis]),
                name="e_step/it_%d/o_p" % it)
        zz = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
                lambda nn, hh, ww, kx, ky, ci, co: tvm.log(o_activations[nn, hh, ww, co] + epsilon) + o_p[nn, hh, ww, kx, ky, ci, co],
                name="e_step/it_%d/zz" % it)
        rr = topi.nn.softmax(zz)
        return rr

    ''' @param poses: shape(n, i_h, i_w, i_size, pose_size, pose_size)
        @param activations: shape(n, i_h, i_w, i_size)
        @param w: shape(kernel_size, kernel_size, i_size, o_size, pose_size, pose_size)
        @param beta_v: shape(o_size)
        @param beta_a: shape(o_size) '''
    def capsule(self, poses, activations, w, beta_v, beta_a, n, i_h, i_w, i_size, o_size, kernel_size, strides, iterations, pose_size):
        o_h = (i_h - kernel_size + 1) // strides
        o_w = (i_w - kernel_size + 1) // strides

        # Tile as convolution input
        poses = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, pose_size, pose_size),
                lambda nn, hh, ww, kx, ky, cc, px, py: poses[nn, hh * strides + kx, ww * strides + ky, cc, px, py],
                name="poses")
        activations = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size),
                lambda nn, hh, ww, kx, ky, cc: activations[nn, hh * strides + kx, ww * strides + ky, cc],
                name="activations")

        # Votes
        vote_k = tvm.reduce_axis((0, pose_size))
        votes = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size, pose_size, pose_size),
                lambda nn, hh, ww, kx, ky, ci, co, px, py: tvm.sum(
                    poses[nn, hh, ww, kx, ky, ci, px, vote_k] * w[kx, ky, ci, co, vote_k, py],
                    axis=[vote_k]),
                name="vote")

        # Routing
        rr = self.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
                lambda nn, hh, ww, kx, ky, ci, co: 1.0 / o_size,
                name="rr")
        it_min = 1.0
        it_max = min(iterations, 3.0)
        for it in range(iterations):
            inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
            o_mean, o_stdv, o_activations = self.m_step(
                    it, rr, votes, activations, beta_v, beta_a, inverse_temperature,
                    n, o_h, o_w, i_size, o_size, kernel_size, pose_size)
            if it < iterations - 1:
                rr = self.e_step(it, o_mean, o_stdv, o_activations, votes, n, o_h, o_w, i_size, o_size, kernel_size, pose_size)
        return o_mean, o_activations

@register_capsule_impl("llvm")
class CapsuleLLVM(Capsule):
    def capsule_schedule(self, o_poses, o_activations, iterations):
        s = tvm.create_schedule([o_poses.op, o_activations.op])
        return s

@register_capsule_impl("cuda")
class CapsuleCUDA(Capsule):
    def capsule_schedule(self, o_poses, o_activations, iterations):
        s = tvm.create_schedule([o_poses.op, o_activations.op])

        def bind_first_2(tensor):
            nn, hh = tensor.op.axis[:2]
            s[tensor].bind(nn, tvm.thread_axis("blockIdx.x"))
            s[tensor].bind(hh, tvm.thread_axis("threadIdx.x"))

        bind_first_2(self.find("poses"))
        bind_first_2(self.find("activations"))
        bind_first_2(self.find("vote"))
        bind_first_2(self.find("rr"))
        for it in range(iterations):
            bind_first_2(self.find("m_step/it_%d/rr_prime" % it))
            bind_first_2(self.find("m_step/it_%d/rr_prime_sum" % it))
            bind_first_2(self.find("m_step/it_%d/o_mean_1" % it))
            bind_first_2(self.find("m_step/it_%d/o_mean_2" % it))
            bind_first_2(self.find("m_step/it_%d/o_stdv_1" % it))
            bind_first_2(self.find("m_step/it_%d/o_stdv_2" % it))
            bind_first_2(self.find("m_step/it_%d/o_cost_h" % it))
            bind_first_2(self.find("m_step/it_%d/o_cost" % it))
            bind_first_2(self.find("m_step/it_%d/o_cost_mean_1" % it))
            bind_first_2(self.find("m_step/it_%d/o_cost_mean_2" % it))
            bind_first_2(self.find("m_step/it_%d/o_cost_stdv_1" % it))
            bind_first_2(self.find("m_step/it_%d/o_cost_stdv_2" % it))
            bind_first_2(self.find("m_step/it_%d/o_activations" % it))
        for it in range(iterations - 1):
            bind_first_2(self.find("e_step/it_%d/o_p" % it))
            bind_first_2(self.find("e_step/it_%d/zz" % it))

        return s

