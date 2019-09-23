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

Var x("x"), y("y"), 
    dx("dx"), dy("dy"),
    ci("ci"), c("c"), n("n");

template <typename InputBuffer, typename OutputBuffer>
std::map<std::string, Func> scatter2gather(
        const InputBuffer &weights,
        const OutputBuffer &output)
{
    Func f_weights("f_weights");
    f_weights(x, y, dx, dy, n) = Halide::BoundaryConditions::constant_exterior(
        weights, 0.0f)(x, y, dx, dy, n);

    Expr kw = weights.dim(2).extent();
    Expr kh = weights.dim(3).extent();

    Expr ddx = dx - (kw-1)/2;
    Expr ddy = dy - (kh-1)/2;

    output(x, y, dx, dy, n) = f_weights(
            x + ddx,
            y + ddy,
            kw-1 - dx,
            kh-1 - dy, n);

    std::map<std::string, Func> func_map;

    return func_map;
}

namespace rendernet {

/**
 * Converts sample-centered kernels into pixel-centered kernels.
 */
class Scatter2GatherGenerator : public Generator<Scatter2GatherGenerator> {
public:
    Input<Buffer<float>> weights{"weights", 5};
    Output<Buffer<float>> output{"output", 5};

    void generate() {
        std::map<std::string, Func> funcs = scatter2gather(
            weights, output);

        Var tx("tx"), ty("ty"), tz("tz"), dxdy("dxdy"),
            xy("xy"), cn("cn"), allvars("allvars");

        if(get_target().has_gpu_feature()) {
            output
                .compute_root()
                // .gpu_tile(x, y, tx, tx, 32, 32)
                .fuse(x, y, xy)
                .fuse(dx, dy, dxdy)
                .fuse(xy, dxdy, allvars)
                .fuse(allvars, n, allvars)
                .gpu_tile(allvars, tx, 1024)
                ;
        } else {
            output
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

HALIDE_REGISTER_GENERATOR(rendernet::Scatter2GatherGenerator, scatter2gather)
