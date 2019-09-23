// Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
// Michaël Gharbi Tzu-Mao Li Miika Aittala Jaakko Lehtinen Frédo Durand
// Siggraph 2019
//
// Copyright (c) 2019 Michaël Gharbi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <map>
#include <string>
#include <Halide.h>


using namespace Halide;

Var x("x"), y("y"), dx("dx"), dy("dy"), ci("ci"), c("c"), n("n");

template <typename InputBuffer, typename OutputBuffer>
std::map<std::string, Func> kernel_weighting(
        const InputBuffer &data,
        const InputBuffer &weights,
        const OutputBuffer &output,
        const OutputBuffer &sum_w)
{
    Func f_data("f_data");
    f_data(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        data, 0.0f)(x, y, ci, n);
    Func f_weights("f_weights");
    f_weights(x, y, dx, dy, n) = Halide::BoundaryConditions::constant_exterior(
        weights, 0.0f)(x, y, dx, dy, n);

    Expr kw = weights.dim(2).extent();
    Expr kh = weights.dim(3).extent();
    Expr channels = data.dim(2).extent();

    RDom r_kernel(0, kw, 0, kh);
    Expr w = f_weights(x, y, r_kernel.x, r_kernel.y, n);

    Func homogeneous("homegeneous");
    homogeneous(x, y, c, n) = select(c < channels, f_data(x, y, c, n), 1.0f);

    Func summed("summed");
    summed(x, y, c, n) = 0.0f;
    summed(x, y, c, n) += w * homogeneous(x + r_kernel.x - (kw-1)/2,
                                          y + r_kernel.y - (kh-1)/2, c, n);

    output(x, y, c, n) = summed(x, y, c, n);
    sum_w(x, y, n) = summed(x, y, channels, n);
    
    std::map<std::string, Func> func_map;

    func_map["summed"] = summed;

    return func_map;
}


template <typename InputBuffer, typename OutputBuffer>
std::map<std::string, Func> kernel_weighting_grad(
        const InputBuffer &data,
        const InputBuffer &weights,
        const InputBuffer &sum_w,
        const InputBuffer &d_output,
        const InputBuffer &d_sum_w,
        const OutputBuffer &d_data,
        const OutputBuffer &d_weights)
{
    Func f_data("f_data");
    f_data(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        data, 0.0f)(x, y, ci, n);
    Func f_d_output("f_d_output");
    f_d_output(x, y, c, n) = Halide::BoundaryConditions::constant_exterior(
        d_output, 0.0f)(x, y, c, n);
    Func f_weights("f_weights");
    f_weights(x, y, dx, dy, n) = Halide::BoundaryConditions::constant_exterior(
        weights, 0.0f)(x, y, dx, dy, n);

    Expr kw = weights.dim(2).extent();
    Expr kh = weights.dim(3).extent();
    Expr channels = data.dim(2).extent();

    RDom r_kernel(0, kw, 0, kh);
   
    Expr w = f_weights(x + r_kernel.x - (kw-1)/2, 
                       y + r_kernel.y - (kh-1)/2,
                       kw - 1 - r_kernel.x,
                       kh - 1 - r_kernel.y, n);

    // out = sum { data * w }
    // dL / ddata = sum {dL/dout * dout / ddata } (= sum {dL/dout * w})
    //              + sum {dL/dsumw * dsumw / ddata} (=0)
    Func d_data_tmp("d_data_tmp");
    d_data_tmp(x, y, c, n) = 0.0f;
    d_data_tmp(x, y, c, n) += w * f_d_output(
            x + r_kernel.x - (kw-1)/2, y + r_kernel.y - (kh-1)/2, c, n);
    d_data(x, y, c, n) = d_data_tmp(x, y, c, n);

    // sumw = sum { w }
    // dL / dwj = sum { dL/dout * dout / dwj } (=sum{dL/dout * dataj})
    //          + sum { dL/dsumw * dsumw / dwj } (=sum{dL/dsumw * wj})
    // Expr w2 = f_weights(x, y, dx, dy, n);
    Func d_weights_tmp("d_weights_tmp");
    RDom rchan(0, data.dim(2).extent());
    d_weights_tmp(x, y, dx, dy, n) = d_sum_w(x, y, n);
    d_weights_tmp(x, y, dx, dy, n) += 
        f_data( x + dx - (kw-1)/2, y + dy - (kh-1)/2, rchan, n)
        * f_d_output(x, y, rchan, n);
    d_weights(x, y, dx, dy, n) = d_weights_tmp(x, y, dx, dy, n);
    
    std::map<std::string, Func> func_map;
    // func_map["d_data_tmp"] = d_data_tmp;
    // func_map["d_weights_tmp"] = d_weights_tmp;

    return func_map;
}

namespace rendernet {

class KernelWeightingForwardGenerator : public Generator<KernelWeightingForwardGenerator> {
public:
    Input<Buffer<float>> data{"data", 4};
    Input<Buffer<float>> weights{"weights", 5};
    Output<Buffer<float>> output{"output", 4};
    Output<Buffer<float>> sum_w{"sum_w", 3};

    void generate() {
        std::map<std::string, Func> funcs = kernel_weighting(
            data, weights, output, sum_w);

        Var tx("tx"), ty("ty"), tz("tz"),
            xy("xy"), cn("cn"), allvars("allvars");
        int ts = 16;


        if(get_target().has_gpu_feature()) {
            output
                .fuse(x, y, xy)
                .fuse(c, n, cn)
                .fuse(xy, cn, allvars)
                .gpu_tile(allvars, tx, 1024, TailStrategy::GuardWithIf)
                ;

            sum_w
                .fuse(x, y, xy)
                .fuse(xy, n, allvars)
                .gpu_tile(allvars, tx, 1024, TailStrategy::GuardWithIf)
                ;

            funcs["summed"]
                .compute_root()
                .gpu_tile(x, y, tx, ty, ts, ts, TailStrategy::GuardWithIf)
                .update()
                .gpu_tile(x, y, tx, ty, ts, ts, TailStrategy::GuardWithIf)
                ;
        } else {
            output
                .compute_root()
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;

            sum_w
                .compute_root()
                .fuse(y, n, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;

            funcs["summed"]
                .compute_root()
                .parallel(y, 8)
                .vectorize(x, 8)
                .update()
                .parallel(y, 8)
                .vectorize(x, 8)
                ;
        }
    }

};

class KernelWeightingGradGenerator : public Generator<KernelWeightingGradGenerator> {
public:
    Input<Buffer<float>> data{"data", 4};
    Input<Buffer<float>> weights{"weights", 5};
    Input<Buffer<float>> sum_w{"sum_w", 3};
    Input<Buffer<float>> d_output{"d_output", 4};
    Input<Buffer<float>> d_sum_w{"d_sum_w", 3};

    Output<Buffer<float>> d_data{"d_data", 4};
    Output<Buffer<float>> d_weights{"d_weights", 5};

    void generate() {
        std::map<std::string, Func> funcs = kernel_weighting_grad(
            data, weights, sum_w, d_output, d_sum_w, d_data, d_weights);


        Var tx("tx"), ty("ty"), tz("tz"), dxdy("dxdy"),
            xy("xy"), cn("cn"), allvars("allvars");

        if(get_target().has_gpu_feature()) {
            d_data
                .gpu_tile(x, y, tx, ty, 32, 32, TailStrategy::GuardWithIf)
                ;
            d_weights
                .gpu_tile(x, y, tx, ty, 32, 32, TailStrategy::GuardWithIf)
                ;
        } else {
            d_data
                .compute_root()
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;

            d_weights
                .compute_root()
                .fuse(dx, dy, dxdy)
                .fuse(y, dxdy, allvars)
                .fuse(allvars, n, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;
        }
    }
};


}  // end namespace rendernet

HALIDE_REGISTER_GENERATOR(
        rendernet::KernelWeightingForwardGenerator, kernel_weighting)

HALIDE_REGISTER_GENERATOR(
        rendernet::KernelWeightingGradGenerator, kernel_weighting_grad)
