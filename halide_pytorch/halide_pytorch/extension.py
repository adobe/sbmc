import os
import platform
import subprocess
import setuptools
from torch.utils.cpp_extension import (BuildExtension, include_paths, library_paths)


class HalideOp(object):
    def __init__(self, gen_source, name, function_name, cuda=False):
        self.gen_source = gen_source
        # gen_name
        self.name = name

        # function name
        self.function_name = function_name

        self.cuda = cuda

        # target
        # autoschedule yes/no
        # other options?

    def __repr__(self):
        return "HalideOp %s -> %s (%s)" % (self.name, self.function_name, self.gen_source)


class HalidePyTorchExtension(setuptools.Extension):
    """A Halide extension for PyTorch.

    Args:
        halide_root(str): path to the root of the Halide distribution.
        name(str): name of the extension library.
        extra_sources(list of str): additional cpp source files to compile with the
            extension.
        generators(list of HalideOp): Halide PyTorch operators to compile and
            add to the extension:
        gen_cxx(str): name of the C++ compiler to use for the Halide generators.
        gen_cxxflags(list of str): C++ compiler flags for the Halide generators.
    """
    def __init__(self, halide_root, name, *args, generators=[], 
                 gen_cxx="g++", gen_cxxflags=None, extra_sources=[], **kwargs):
        sources = extra_sources
        cuda = False
        for g in generators:
            # Activate cuda in the wrapper whenever we have an op that requires it
            cuda = cuda or g.cuda

        print("CUDA?", cuda)

        compile_args = kwargs.get('extra_compile_args', [])
        compile_args += ["-std=c++11", "-g"]
        if platform.system() == "Darwin":  # on osx libstdc++ causes trouble
            compile_args += ["-stdlib=libc++"]  
        kwargs["extra_compile_args"] = compile_args

        include_dirs = kwargs.get('include_dirs', [])
        library_dirs = kwargs.get('library_dirs', [])
        libraries = kwargs.get('libraries', [])

        include_dirs += include_paths(cuda=cuda)
        include_dirs.append(os.path.join(halide_root, "include"))

        if cuda:
            libraries.append('cudart')
            libraries.append('cuda')

        if platform.system() == 'Windows':
            library_dirs += library_paths()
            kwargs['library_dirs'] = library_dirs

            libraries = kwargs.get('libraries', [])
            libraries.append('c10')
            if cuda:
                libraries.append('c10_cuda')
            libraries.append('torch')
            libraries.append('torch_python')
            libraries.append('_C')
            kwargs['libraries'] = libraries

        kwargs['language'] = 'c++'

        if cuda:
            library_dirs += library_paths(cuda=True)

        kwargs['include_dirs'] = include_dirs
        kwargs['library_dirs'] = library_dirs
        kwargs['libraries'] = libraries

        super(HalidePyTorchExtension, self).__init__(name, sources, *args, **kwargs)

        # Group generators by source file, so we compile those only once
        self.generators = {}
        self.cuda = cuda
        for g in generators:
            if not g.gen_source in self.generators.keys():
                self.generators[g.gen_source] = []
            self.generators[g.gen_source].append(g)

        self.gen_cxx = gen_cxx
        self.gen_cxxflags = self._get_gen_cxxflags(gen_cxxflags, halide_root)
        self.gen_ldflags = self._get_gen_ldflags(None)
        self.gen_hlsyslibs = self._get_hlsyslibs(None)
        self.gen_deps = self._get_gen_deps(None, halide_root)

    def __repr__(self):
        return "HalidePyTorchExtension"

    def _get_gen_cxxflags(self, flags, hl_distrib):
        if flags is None:
            flags =["-O3", "-std=c++11",
                    "-I", os.path.join(hl_distrib, "include"),
                    "-I", os.path.join(hl_distrib, "tools"),
                    "-Wall", "-Werror", "-Wno-unused-function", 
                    "-Wcast-qual", "-Wignored-qualifiers",
                    "-Wno-comment", "-Wsign-compare",
                    "-Wno-unknown-warning-option",
                    "-Wno-psabi"]
        return flags

    def _get_gen_ldflags(self, flags):
        if flags is None:
            flags = ["-ldl", "-lpthread", "-lz"]
        return flags

    def _get_hlsyslibs(self, flags):
        if flags is None:
            # TODO: load from distrib Make config
            if platform.system() == 'Darwin':
                flags = ["-lz", "-lxml2", "-lm"]
            else:  # Linux
                flags = ["-lz", "-lrt", "-ldl", "-ltinfo", "-lpthread", "-lm"]
        return flags

    def _get_gen_deps(self, flags, hl_distrib):
        if platform.system() == 'Darwin':
            ext = ".dylib"
        else:
            ext = ".so"

        if flags is None:
            flags = [os.path.join(hl_distrib, "bin", "libHalide"+ext),
                     # os.path.join(hl_distrib, "include", "Halide.h"),
                     os.path.join(hl_distrib, "tools", "GenGen.cpp")]

        return flags


class HalideBuildExtension(BuildExtension):
    def _generate_pybind_wrapper(self, path, headers, cuda):
        """Synthesizes a .cpp source file with a PyBind wrapper around the Halide
        ops.

        Args:
            path(str): full path for the synthesized .cpp file.
            headers(list of str): list of paths to the Halide PyTorch extension 
                headers to include and wrap.
            cuda(bool): if True, include cuda headers.
        """
        s = "#include \"torch/extension.h\"\n\n"
        if cuda:
            s += "#define HL_PT_CUDA\n"
        s += "#include \"HalidePyTorchHelpers.h\"\n"
        for h in headers:
            s += "#include \"{}\"\n".format(os.path.splitext(h)[0]+".pytorch.h")
        if cuda:
            s += "#undef HL_PT_CUDA\n"

        s += "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        for h in headers:
            name = os.path.splitext(h)[0]
            s += "  m.def(\"{}\", &{}_th_, \"PyTorch wrapper of the Halide pipeline {}\");\n".format(
              name, name, name)
        s += "}\n"
        with open(path, 'w') as fid:
            fid.write(s)

    def run(self):
        if platform.system() == 'Windows':
            raise RuntimeError("windows is not supported currently.")

        try:
            out = subprocess.check_output(['make', '--version'])
        except OSError:
            raise RuntimeError(
                "Make must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        super(HalideBuildExtension, self).run()


    def build_extensions(self):
        exts = self.extensions

        # Create temporary build directory
        build = self.build_temp
        os.makedirs(build, exist_ok=True)

        hl_libs = []
        hl_headers = []
        cuda = False
        for ext in exts:
            if isinstance(ext, HalidePyTorchExtension):
                cxx = [ext.gen_cxx]
                cxxflags = ext.gen_cxxflags
                ldflags = ext.gen_ldflags
                hlsyslibs = ext.gen_hlsyslibs
                gendeps = ext.gen_deps

                generators = ext.generators
                print("building Halide PyTorch extension with generators:", generators)

                # Activate cuda in the wrapper whenever we have an ext that requires it
                cuda = cuda or ext.cuda

                for g in generators:
                    # TODO: test generator names are unique and key is a .cpp file
                    generator_id = os.path.basename(os.path.splitext(g)[0])
                    generator_bin = os.path.join(build, "%s.generator" % generator_id)

                    cmd = cxx + cxxflags + [g] + gendeps + ["-o", generator_bin] + ldflags + hlsyslibs

                    print("building generator", generator_bin)
                    subprocess.check_call(cmd)

                    for gen in generators[g]:
                        hl_lib = os.path.join(build)
                        env = os.environ.copy()
                        op_cuda = gen.cuda
                        target = "target=host"
                        if op_cuda:
                            target += "-cuda-cuda_capability_61-user_context"
                        # TODO: add linux version
                        env["DYLD_LIBRARY_PATH"] = "../Halide/bin"
                        cmd2 =  [generator_bin, "-g", gen.name, "-f",
                                 gen.function_name, "-e",
                                 "static_library,h,pytorch_wrapper",
                                 "-o", hl_lib, target,
                                 "auto_schedule=False"]
                        libname = os.path.join(hl_lib, gen.function_name+".a")
                        header = gen.function_name+".h"
                        hl_libs.append(libname)
                        hl_headers.append(header)
                        print("building halide operator %s" % (libname))
                        subprocess.check_call(cmd2, env=env)

        wrapper_path = os.path.join(build, "wrapper.cpp")
        self._generate_pybind_wrapper(wrapper_path, hl_headers, cuda)

        for ext in self.extensions:
            ext.extra_objects += hl_libs  # add the static op libraries
            ext.sources.append(wrapper_path)  # and the wrapper's source

        super(HalideBuildExtension, self).build_extensions()


