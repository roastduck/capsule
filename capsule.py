import tvm
import topi

epsilon = 1e-5

def square(x):
    return x * x

''' @param rr: shape(n, o_h, o_w, kernel_size, kernel_size, i_size, o_size)
    @param votes: shape(n, o_h, o_w, kernel_size, kernel_size, i_size, o_size, pose_size, pose_size)
    @param i_activations: shape(n, o_h, o_w, kernel_size, kernel_size, i_size)
    @param beta_v: shape(o_size)
    @param beta_a: shape(o_size) '''
def m_step(it, rr, votes, i_activations, beta_v, beta_a, inverse_temperature, n, o_h, o_w, i_size, o_size, kernel_size, pose_size):
    rr_prime = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
            lambda nn, hh, ww, kx, ky, ci, co: rr[nn, hh, ww, kx, ky, ci, co] * i_activations[nn, hh, ww, kx, ky, ci])

    kx_axis, ky_axis, ci_axis = map(tvm.reduce_axis, [(0, kernel_size), (0, kernel_size), (0, i_size)])
    rr_prime_sum = tvm.compute((n, o_h, o_w, o_size),
            lambda nn, hh, ww, co: tvm.sum(
                rr_prime[nn, hh, ww, kx_axis, ky_axis, ci_axis, co],
                axis=[kx_axis, ky_axis, ci_axis]),
            name="m_step/it_%d/rr_prime_sum" % it)

    kx_axis, ky_axis, ci_axis = map(tvm.reduce_axis, [(0, kernel_size), (0, kernel_size), (0, i_size)])
    o_mean = tvm.compute((n, o_h, o_w, o_size, pose_size, pose_size),
            lambda nn, hh, ww, co, px, py: tvm.sum(
                rr_prime[nn, hh, ww, kx_axis, ky_axis, ci_axis, co] * votes[nn, hh, ww, kx_axis, ky_axis, ci_axis, co, px, py],
                axis=[kx_axis, ky_axis, ci_axis]),
            name="m_step/it_%d/o_mean_1" % it)
    o_mean = tvm.compute((n, o_h, o_w, o_size, pose_size, pose_size),
            lambda nn, hh, ww, co, px, py: o_mean[nn, hh, ww, co, px, py] / rr_prime_sum[nn, hh, ww, co],
            name="m_step/it_%d/o_mean_2" % it)

    kx_axis, ky_axis, ci_axis = map(tvm.reduce_axis, [(0, kernel_size), (0, kernel_size), (0, i_size)])
    o_stdv = tvm.compute((n, o_h, o_w, o_size, pose_size, pose_size),
            lambda nn, hh, ww, co, px, py: tvm.sum(
                rr_prime[nn, hh, ww, kx_axis, ky_axis, ci_axis, co] * square(
                    votes[nn, hh, ww, kx_axis, ky_axis, ci_axis, co, px, py] - o_mean[nn, hh, ww, co, px, py]),
                axis=[kx_axis, ky_axis, ci_axis]),
            name="m_step/it_%d/o_stdv_1" % it)
    o_stdv = tvm.compute((n, o_h, o_w, o_size, pose_size, pose_size),
            lambda nn, hh, ww, co, px, py: tvm.sqrt(o_stdv[nn, hh, ww, co, px, py]) / rr_prime_sum[nn, hh, ww, co],
            name="m_step/it_%d/o_stdv_2" % it)

    o_cost_h = tvm.compute((n, o_h, o_w, o_size, pose_size, pose_size),
            lambda nn, hh, ww, co, px, py: (beta_v[co] + tvm.log(o_stdv[nn, hh, ww, co, px, py] + epsilon)) * rr_prime_sum[nn, hh, ww, co])

    px_axis, py_axis, = map(tvm.reduce_axis, [(0, pose_size), (0, pose_size)])
    o_cost = tvm.compute((n, o_h, o_w, o_size),
            lambda nn, hh, ww, co: tvm.sum(
                o_cost_h[nn, hh, ww, co, px_axis, py_axis],
                axis=[px_axis, py_axis]))

    co_axis = tvm.reduce_axis((0, o_size))
    o_cost_mean = tvm.compute((n, o_h, o_w),
            lambda nn, hh, ww: tvm.sum(
                o_cost[nn, hh, ww, co_axis],
                axis=[co_axis]))
    o_cost_mean = tvm.compute((n, o_h, o_w),
            lambda nn, hh, ww: o_cost_mean[nn, hh, ww] / o_size)

    co_axis = tvm.reduce_axis((0, o_size))
    o_cost_stdv = tvm.compute((n, o_h, o_w),
            lambda nn, hh, ww: tvm.sum(
                square(o_cost[nn, hh, ww, co_axis] - o_cost_mean[nn, hh, ww]),
                axis=[co_axis]))
    o_cost_stdv = tvm.compute((n, o_h, o_w),
            lambda nn, hh, ww: tvm.sqrt(o_cost_stdv[nn, hh, ww] / o_size))
    o_activations = tvm.compute((n, o_h, o_w, o_size),
            lambda nn, hh, ww, co: tvm.sigmoid(
                (beta_a[co] + (o_cost_mean[nn, hh, ww] - o_cost[nn, hh, ww, co]) / (o_cost_stdv[nn, hh, ww] + epsilon)) * inverse_temperature))

    return o_mean, o_stdv, o_activations

''' @param o_mean: shape(n, o_h, o_w, o_size, pose_size, pose_size)
    @param o_stdv: shape(n, o_h, o_w, o_size, pose_size, pose_size)
    @param o_activations: shape(n, o_h, o_w, o_size)
    @param votes: shape(n, o_h, o_w, kernel_size, kernel_size, i_size, o_size, pose_size, pose_size) '''
def e_step(it, o_mean, o_stdv, o_activations, votes, n, o_h, o_w, i_size, o_size, kernel_size, pose_size):
    px_axis, py_axis, = map(tvm.reduce_axis, [(0, pose_size), (0, pose_size)])
    o_p = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
            lambda nn, hh, ww, kx, ky, ci, co: tvm.sum(
                -square(votes[nn, hh, ww, kx, ky, ci, co, px_axis, py_axis] - o_mean[nn, hh, ww, co, px_axis, py_axis]) /
                    (2 * square(o_stdv[nn, hh, ww, co, px_axis, py_axis]))
                -tvm.log(o_stdv[nn, hh, ww, co, px_axis, py_axis] + epsilon),
                axis=[px_axis, py_axis]))
    zz = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
            lambda nn, hh, ww, kx, ky, ci, co: tvm.log(o_activations[nn, hh, ww, co] + epsilon) + o_p[nn, hh, ww, kx, ky, ci, co])
    rr = topi.nn.softmax(zz)
    return rr

''' @param poses: shape(n, i_h, i_w, i_size, pose_size, pose_size)
    @param activations: shape(n, i_h, i_w, i_size)
    @param w: shape(kernel_size, kernel_size, i_size, o_size, pose_size, pose_size)
    @param beta_v: shape(o_size)
    @param beta_a: shape(o_size) '''
def capsule(poses, activations, w, beta_v, beta_a, n, i_h, i_w, i_size, o_size, kernel_size, strides, iterations, pose_size):
    o_h = (i_h - kernel_size + 1) // strides
    o_w = (i_w - kernel_size + 1) // strides

    # Tile as convolution input
    poses = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, pose_size, pose_size),
            lambda nn, hh, ww, kx, ky, cc, px, py: poses[nn, hh * strides + kx, ww * strides + ky, cc, px, py])
    activations = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size),
            lambda nn, hh, ww, kx, ky, cc: activations[nn, hh * strides + kx, ww * strides + ky, cc])

    # Votes
    vote_k = tvm.reduce_axis((0, pose_size))
    votes = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size, pose_size, pose_size),
            lambda nn, hh, ww, kx, ky, ci, co, px, py: tvm.sum(
                poses[nn, hh, ww, kx, ky, ci, px, vote_k] * w[kx, ky, ci, co, vote_k, py],
                axis=[vote_k]),
            name="vote")

    # Routing
    rr = tvm.compute((n, o_h, o_w, kernel_size, kernel_size, i_size, o_size),
            lambda nn, hh, ww, kx, ky, ci, co: 1.0 / o_size)
    it_min = 1.0
    it_max = min(iterations, 3.0)
    for it in range(iterations):
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
        o_mean, o_stdv, o_activations = m_step(
                it, rr, votes, activations, beta_v, beta_a, inverse_temperature,
                n, o_h, o_w, i_size, o_size, kernel_size, pose_size)
        if it < iterations - 1:
            rr = e_step(it, o_mean, o_stdv, o_activations, votes, n, o_h, o_w, i_size, o_size, kernel_size, pose_size)
    return o_mean, o_activations

def capsule_schedule(o_poses, o_activations):
    s = tvm.create_schedule([o_poses.op, o_activations.op])
    return s

