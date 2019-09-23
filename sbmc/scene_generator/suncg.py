"""This file contains the scene generator we used to interface we SunCG.

Given the issues with SunCG discovered in 2019, the code in this file is no longer
supported nor maintained.
"""
class TextureParser(object):
    def __init__(self):
        super(TextureParser, self).__init__()
        self.re = re.compile(
            r'^Texture\s*"(?P<name>[^"]*)"\s*"(?P<type>[^"]*)"\s*"(?P<tex>[^"]*)"(?P<rest>.*)')
        self.imagemap_re = re.compile(
            r'.*"string\s*filename"\s*\[\s*"(?P<filename>[^"]*)"\s*\].*')
        self.match_ = None

    def match(self, s):
        self.match_ = self.re.match(s)
        return self.match_

    def parse(self, s):
        if self.match_ is None:
            raise ValueError("no match!")

        name = self.match_.group("name")
        type = self.match_.group("type")
        tex = self.match_.group("tex")
        rest = self.match_.group("rest")

        if tex == "imagemap":
            m = self.imagemap_re.match(rest)
            if m:
                return Imagemap(name, type, m.group("filename"))
            else:
                raise ValueError("Could not match filename")
        elif tex == "constant":
            return
        elif tex == "scale":
            return
        else:
            raise ValueError("unrecognized texture {}".format(tex))

        self.match_ = None


class SuncgConverter():
    def __init__(self, suncg_root):
        self.gaps = os.path.join(
            suncg_root, "toolkit", "gaps", "bin", "x86_64")
        self.houses = os.path.join(suncg_root, "house")
        self.cameras = os.path.join(suncg_root, "cameras")
        self.objects = os.path.join(suncg_root, "object")
        self.rooms = os.path.join(suncg_root, "room")
        self.classes_file = os.path.join(suncg_root, "pbrs", "util_data",
                                         "ModelCategoryMappingNewActive.csv")
        self.light_materials_file = os.path.join(suncg_root, "pbrs", "util_data",
                                                 "light_geometry_compact.txt")
        project_list = os.path.join(suncg_root, "project_ids.txt")

        self.classes = self._load_classes(self.classes_file)
        self.light_materials = self._load_light_materials(
            self.light_materials_file)

        with open(project_list) as fid:
            plist = [f.strip() for f in fid.readlines()]
        np.random.shuffle(plist)
        self.plist = plist

    def convert_room(self, scene, rm, dst_dir, pbrt_converter):
        if rm["valid"] != 1:
            print("invalid object")

        objects = []
        for ext in ["c", "f", "w"]:
            mdl_name = rm["modelId"] + ext
            obj_file = os.path.join(self.rooms, scene, mdl_name)
            mtl_file = os.path.basename(obj_file + ".mtl")
            obj_file += ".obj"
            if not os.path.exists(obj_file):
                raise ValueError(
                    "Room file does no exists {}".format(obj_file))
            obj_ = self._convert_obj(
                obj_file, mtl_file, dst_dir, pbrt_converter)
            objects += obj_
        return objects

    def convert_object(self, obj, dst_dir, pbrt_converter):
        mdl_id = obj["modelId"]
        obj_dir = os.path.join(self.objects, mdl_id)
        mtl_file = mdl_id + ".mtl"

        if "state" in obj.keys():
            if obj["state"] != 0:
                mdl_id += "_{}".format(obj["state"]-1)

        obj_file = os.path.join(obj_dir, mdl_id + ".obj")

        objects = self._convert_obj(
            obj_file, mtl_file, dst_dir, pbrt_converter)
        for o in objects:
            o["transform"] = obj["transform"]

        return objects

    def _convert_obj(self, obj_file, mtl_file, dst_dir, pbrt_converter):
        basename = os.path.basename(obj_file)
        dirname = os.path.dirname(obj_file)
        dst = os.path.join(dst_dir, basename)
        _obj_split_material_groups(obj_file, dst)

        pbrt_file = os.path.splitext(basename)[0] + ".pbrt"

        cwd = os.getcwd()
        os.chdir(dst_dir)
        if not os.path.exists(mtl_file):
            os.symlink(os.path.join(dirname, mtl_file), mtl_file)
        cmd = [pbrt_converter, basename, pbrt_file]
        subprocess.call(cmd)
        objects = self.rewrite_pbrt(pbrt_file)
        os.remove(basename)
        os.remove(mtl_file)
        os.chdir(cwd)

        return objects

    def rewrite_pbrt(self, in_f):
        """"""
        tmp_f = in_f+".rewrite"

        object_idx = 0
        objects = []

        objre = re.compile(r'^# Name\s*"(?P<obj_name>.*)".*$')
        with open(in_f) as fid:
            l = fid.readline()
            while l:
                m = objre.match(l)
                if m:  # We have a new object
                    name = m.group("obj_name")
                    obj, mat = name.split("@")

                    cat = self.get_obj_category(obj, mat)

                    # Remove header
                    while not l.startswith("Material") or l.startswith("Shape"):
                        l = fid.readline()

                    # Find material definition
                    if l.startswith("Material"):
                        mat_ = self._parse_material(l)
                    else:
                        print("!! found no material !!")
                        mat_ = None

                    while not l.startswith("Shape"):
                        l = fid.readline()

                    # Write a new geometry file for this object
                    id = str(uuid.uuid4()).replace("-", "_")
                    new_f = os.path.splitext(
                        in_f)[0] + "{}_object{:04d}.pbrt".format(id, object_idx)
                    obj_ = {}
                    with open(new_f, 'w') as new_fid:
                        # Write new material link
                        new_fid.write('AttributeBegin\n')
                        if mat is not None:
                            new_fid.write(
                                '# Object "{}" Material "{}"\n'.format(name, mat))
                            new_fid.write('NamedMaterial "{}"\n'.format(id))
                        mat_["id"] = id
                        obj_["category"] = cat

                        while not l.strip() == "AttributeEnd":
                            new_fid.write(l)
                            l = fid.readline()
                        new_fid.write(l)

                    object_idx += 1

                    obj_["path"] = new_f
                    obj_["material"] = mat_
                    objects.append(obj_)
                l = fid.readline()
        return objects

    MAT_RE = re.compile(
        r'.*"float roughness"\s\[(?P<roughness>[^\]]*)\]\s.*"float index"\s*\[(?P<index>[^\]]*)\]\s.*"rgb opacity"\s*\[(?P<opacity>[^\]]*)\].*')

    def _parse_material(self, l):
        opm = SuncgConverter.MAT_RE.match(l)
        mat = {
            "roughness": float(opm.group("roughness")),
            "index": float(opm.group("index")),
            "opacity": min([float(c) for c in opm.group("opacity").split()]),
        }
        return mat

    def _load_classes(self, mapping_file):
        window_ids = []
        door_ids = []
        plant_ids = []
        people_ids = []
        mirror_ids = []
        with open(mapping_file) as fid:
            wset = ["window", "windows"]
            pset = ["person", "people"]
            reader = csv.DictReader(fid)
            for row in reader:
                if row["fine_grained_class"] in wset or (
                    row["coarse_grained_class"] in wset or (
                        row["nyuv2_40class"] in wset)):
                    window_ids.append(row["model_id"])
                elif row["fine_grained_class"] == "door" or (
                    row["coarse_grained_class"] == "door" or (
                        row["nyuv2_40class"] == "door")):
                    door_ids.append(row["model_id"])
                elif row["fine_grained_class"] == "plant" or (
                    row["coarse_grained_class"] == "plant" or (
                        row["nyuv2_40class"] == "plant")):
                    plant_ids.append(row["model_id"])
                elif row["fine_grained_class"] in pset or (
                    row["coarse_grained_class"] in pset or (
                        row["nyuv2_40class"] in pset)):
                    people_ids.append(row["model_id"])
                elif row["fine_grained_class"] == "mirror":
                    mirror_ids.append(row["model_id"])
        remove_ids = people_ids + plant_ids
        pure_trans_ids = window_ids + door_ids
        geometry = {"remove": remove_ids, "transparent": pure_trans_ids,
                    "mirror": mirror_ids}
        return geometry

    def _load_light_materials(self, lighting_file):
        bulb_ids = []
        shade_ids = []
        light_models = []
        with open(lighting_file) as fid:
            lno = 0
            for l in fid.readlines():
                lno += 1
                data = l.split()
                mdl = data[0]
                light_models.append(mdl)
                bulb_count = int(data[1])
                i = 2

                cur_bulb_ids = []
                for _ in range(bulb_count):
                    cur_bulb_ids.append(data[i])
                    i += 1
                bulb_ids.append(cur_bulb_ids)

                cur_shade_ids = []
                shade_count = int(data[i])
                i += 1
                for _ in range(shade_count):
                    cur_shade_ids.append(data[i])
                    i += 1
                shade_ids.append(cur_shade_ids)
        lights = {"models": light_models,
                  "bulbs": bulb_ids, "shades": shade_ids}
        return lights

    def load_housedata(self, scene):
        path = os.path.join(self.houses, scene, "house.json")
        with open(path) as fid:
            house_data = json.load(fid)
        return house_data

    def cameras_for_scene(self, scene, shuffle=False):
        good_cams = os.path.join(self.cameras, scene, "room_camera_good.txt")
        if not os.path.exists(good_cams):
            return None

        with open(good_cams) as fid:
            good_cams = [bool(l.strip()) for l in fid.readlines()]

        cam = os.path.join(self.cameras, scene, "room_camera.txt")
        with open(cam) as fid:
            cam = [[float(c) for c in l.strip().split()]
                   for l in fid.readlines()]

        cam_name = os.path.join(self.cameras, scene, "room_camera_name.txt")
        with open(cam_name) as fid:
            for i, l in enumerate(fid.readlines()):
                room = l.strip().split("#")[-1].split("_")[:-1]
                room = "_".join(room)
                cam[i] = {"camera": cam[i], "room": room}

        cam = [cam[i] for i, g in enumerate(good_cams) if g]

        if shuffle:
            np.random.shuffle(cam)

        return cam

    def get_obj_category(self, name, mat):
        if name in self.classes["transparent"]:
            return "transparent"
        elif name in self.classes["mirror"]:
            return "mirror"
        elif name in self.light_materials["models"]:
            idx = self.light_materials["models"].index(name)
            bulb = self.light_materials["bulbs"][idx]
            shade = self.light_materials["shades"][idx]
            if mat in shade:
                return "light_shade"
            elif mat in bulb:
                return "light_bulb"
            else:
                return "shape"
        else:
            return "shape"


class SunCGSynthesizer(Synthesizer):
    def __init__(self, envmaps, textures, models, pbrt_converter, converter):
        super(SunCGSynthesizer, self).__init__(
            envmaps, textures, models, pbrt_converter)
        self._c = converter

    def sample(self, scn, dst_dir, params=None):
        self.randomize_textures()

        # Randomize motion blur and depth of field
        do_dof = np.random.choice([True, False])
        do_mblur = np.random.choice([True, False])

        # print("mblur?", do_mblur)

        scene = self.get_scene()
        try:
            c = self.get_random_viewpoint(scene)
            room_id = c["room"]
            nodes, bbox = self.parse_house(scene, room_id)
            all_objects, room_bbox = self.parse_scene(
                scene, room_id, nodes, dst_dir)
            obj_positions = self.populate_scene(scn, all_objects)
            # self.add_random_lights(scn, room_bbox)
            cam_params, p0v, cam_vec = self.randomize_camera(
                c, do_dof, room_bbox)
            if do_mblur:
                cam_params = self.random_motion_blur(
                    cam_params, scn, room_bbox, dst_dir, p0v, cam_vec)
            scn.camera = Camera(cam_params)

            if do_mblur:
                if scn.camera.params.shutteropen != 0.0 or scn.camera.params.shutterclose != 1.0:
                    return False
            if do_dof:
                if not scn.camera.params.lensradius > 0.0 or not scn.camera.params.focaldistance > 0.0:
                    return False

            return True
        except InvalidSunCGSceneError:
            # print("invalid CAM", scn.camera, do_mblur, do_dof)
            return False

    def get_scene(self):
        return np.random.choice(self._c.plist)

    def get_random_viewpoint(self, scene):
        cams = self._c.cameras_for_scene(scene, shuffle=True)
        if cams is None:
            # print("No good camera found, skipping scene")
            raise InvalidSunCGSceneError()
        cam = np.random.choice(cams)
        return cam

    def parse_house(self, scene, room_id):
        house = self._c.load_housedata(scene)
        nlevels = len(house["levels"])
        lvl = int(room_id.split("_")[0])
        nodes = house["levels"][lvl]["nodes"]
        bbox = house["levels"][lvl]["bbox"]
        return nodes, bbox

    def parse_scene(self, scene, room_id, nodes, dst_dir):
        all_objects = []
        room_bbox = None
        for n in nodes:
            tp = n["type"]
            if tp == "Room":
                if n["id"] != room_id:
                    continue
                indices = n["nodeIndices"]
                room_bbox = n["bbox"]

                objects = self._c.convert_room(
                    scene, n, os.path.join(dst_dir, "geometry"), self.converter.pbrt_converter)

                for o in objects:
                    o["category"] = "room_" + o["category"]

                all_objects += objects

                for idx in indices:
                    obj = nodes[idx]
                    if "modelId" not in obj.keys() or obj['valid'] != 1:
                        continue

                    objects = self._c.convert_object(
                        obj, os.path.join(dst_dir, "geometry"), self.converter.pbrt_converter)

                    all_objects += objects
        return all_objects, room_bbox

    def populate_scene(self, scn, all_objects):
        window_mode = np.random.choice(["keep", "remove", "area_light"])

        if window_mode != "area_light":
            env = random_envmap(self.envmaps, nsamples=8)
            rotate(env, [1, 0, 0], -90)
            rotate(env, [0, 0, 1], np.random.uniform(0, 360))
            scn.lights.append(env)

        nlights = 0
        ignored = 0
        for o in all_objects:
            cat = o["category"]
            if self.is_light(o, window_mode):
                itsty = np.random.uniform(10, 30)
                light = AreaLight(spectrum=[itsty]*3,
                                  geom=ExternalGeometry(os.path.join("geometry", o["path"])))
                xform = np.array(o["transform"])
                transform(light, xform)
                scn.lights.append(light)
                nlights += 1
            elif o["category"] == "transparent" and window_mode == "remove":
                ignored += 1
                nlights += 1
                continue
            elif o["category"] == "light_shade":
                ignored += 1
                continue
            else:
                # Add shape to scene
                geom = ExternalGeometry(os.path.join("geometry", o["path"]))
                if 'transform' in o.keys():
                    xform = np.array(o["transform"])
                    transform(geom, xform)
                scn.shapes.append(geom)

                mat = o["material"]
                if mat:
                    m = None
                    if window_mode == "keep" and cat == "transparent" and mat["opacity"] < 1.0:
                        nlights += 1
                        m = UberMaterial(
                            id=mat["id"], opacity=mat["opacity"],
                            roughness=mat["roughness"], index=mat["index"])
                    elif cat == "mirror":
                        m = MirrorMaterial(
                            id=mat["id"])
                    else:
                        m = random_material(
                            id=mat["id"], textures=self.current_textures)
                    scn.materials.append(m)

        # print("{} objects in scenes, ignored {}".format(len(all_objects), ignored))
        # print("{} lights in scenes".format(nlights))

        if nlights == 0:
            # print("no lights in scene, skipping")
            raise InvalidSunCGSceneError()

    def randomize_camera(self, c, do_dof, room_bbox):
        c = c["camera"]
        p0 = [c[0], c[1], c[2]]
        p1 = [c[0] + c[3], c[1] + c[4], c[2] + c[5]]
        up = [c[6], c[7], c[8]]

        # Randomize up orientation, sometimes
        if np.random.choice([True, False]):
            up = list(np.random.uniform(size=(3,)))

        fov = np.random.uniform(35, 60)

        cam_params = {"position": p0, "target": p1,
                      "up": up, "fov": fov}

        # Auxiliary cam info
        p0v = np.array(p0)
        cam_vec = np.array(p1)-p0v

        if do_dof:
            tgt_p = None
            while tgt_p is None:
                tgt_p = self.sample_point_in_room(room_bbox)
                fdist = np.dot(tgt_p-p0v, cam_vec)
                if fdist < 1:  # Rejection sample
                    tgt_p = None  # forbid focus points closer than 1m
            tgt_p = np.array(tgt_p)
            # print("camera", p0, p1, tgt_p)
            aperture = _random_aperture()
            cam_params["lensradius"] = aperture
            cam_params["focaldistance"] = fdist
            # print("Using DoF, focus distance:", fdist, "aperture:", aperture)

        return cam_params, p0v, cam_vec

    def random_motion_blur(self, cam_params, scn, room_bbox, dst_dir, p0v, cam_vec):
        cam_params["shutterclose"] = 1.0
        n_objs = np.random.randint(5, 25)
        # print("Adding", n_objs, "objects")
        for obj_id in range(n_objs):
            mdl = np.random.choice(self.models)
            objects = self.converter.convert(
                mdl, os.path.join(dst_dir, "geometry"))

            src_p = None
            while src_p is None:
                src_p = self.sample_point_in_room(room_bbox, margin=0.01)
                fdist = np.dot(src_p-p0v, cam_vec)
                if fdist < 1:  # Rejection sample
                    src_p = None  # forbid focus points close to 1m

            rot = np.random.uniform(0, 360)
            rot_axis = np.random.uniform(size=(3, ))
            rot_axis /= np.linalg.norm(rot_axis)
            rot_axis = list(rot_axis)
            f_scale = list(np.random.uniform(0.5, 3.5)*np.ones((3,)))

            mvec = np.random.uniform(size=(3, ))
            mvec /= np.linalg.norm(mvec)
            distance = np.random.exponential(0.3)
            mvec = list(mvec*distance)

            # print("motion blur", mvec)

            for o in objects:
                geom = ExternalGeometry(os.path.join("geometry", o["path"]))
                scale(geom, f_scale)
                rotate(geom, rot_axis, rot)
                translate(geom, src_p)
                translate(geom, mvec, target="end")
                scn.shapes.append(geom)

                mat = o["material"]
                if mat:
                    m = random_material(
                        id=mat["id"], textures=self.current_textures)
                    scn.materials.append(m)
        return cam_params

    def add_random_lights(self, scn, room_bbox):
        n_added_lights = np.random.randint(1, 10)
        for i in range(n_added_lights):
            light_p = self.sample_point_in_room(room_bbox)
            color = np.random.uniform(0.01, 1, size=(3,))
            color = color / np.linalg.norm(color)
            if np.random.uniform() < .5:
                sz = np.random.uniform(0.01, 0.2)
                geom = Sphere(sz)
                power = np.random.uniform(50, 400) / sz*sz
                power /= n_added_lights
                light = AreaLight(geom, spectrum=list(power*color))
            else:
                power = np.random.uniform(50, 400)
                power /= n_added_lights
                light = PointLight(spectrum=list(power*color))
            translate(light, light_p)
            scn.lights.append(light)

    def is_light(self, o, window_treatment):
        light = "light_bulb" in o["category"]
        win = (window_treatment ==
               "area_light") and o["category"] == "transparent"
        win = win and o["material"]["opacity"] < 1.0
        return light or win

    def sample_point_in_room(self, room_bbox, margin=0.00):
        assert margin >= 0.0 and margin <= 1.0, "margin should be in [0, 1]"
        pos = []
        for i in range(3):
            mini = room_bbox['min'][i]*(1+margin)
            maxi = room_bbox['max'][i]*(1+margin)
            pos.append(np.random.uniform(mini, maxi))
        return np.array(pos)
