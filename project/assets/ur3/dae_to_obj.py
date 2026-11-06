"""One-shot COLLADA -> OBJ converter for the ur_description UR3 visual meshes.

Each DAE holds several sub-meshes, one per material (grey shell, blue plastic,
black joint band, ...). MuJoCo cannot read DAE and one OBJ carries one material,
so each link is split into <link>_<n>.obj, the same layout MuJoCo Menagerie uses.

The OBJ files next to this script were produced from the DAE visuals that ship
in ur_description (github.com/UniversalRobots/Universal_Robots_ROS2_Description),
under meshes/ur3/visual. They are not in this repo any more, so fetch them from
the commit that still had them and point the script at those:

    mkdir -p /tmp/ur3_dae
    for f in base shoulder upperarm forearm wrist1 wrist2 wrist3; do
        git show 88e3d88:Robots/meshes/ur3/visual/$f.dae > /tmp/ur3_dae/$f.dae
    done
    python dae_to_obj.py /tmp/ur3_dae .

Only rerun it if the meshes need regenerating; the output is committed. The
parts come out in the link frame the URDF uses, which is the same frame as the
collision hulls in collision/, so the geom pos/quat in ur3_pick_scene.xml are
shared between the two.
"""
import sys, json
from pathlib import Path
import xml.etree.ElementTree as ET

NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
LINKS = ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]


def tag(e):
    return e.tag.split('}')[-1]


def floats(root, src_id):
    """Resolve a <source>/<vertices> url to its float array and stride."""
    src_id = src_id.lstrip('#')
    v = root.find(f".//c:vertices[@id='{src_id}']", NS)
    if v is not None:
        src_id = v.find("c:input[@semantic='POSITION']", NS).get('source').lstrip('#')
    s = root.find(f".//c:source[@id='{src_id}']", NS)
    arr = s.find('.//c:float_array', NS)
    acc = s.find('.//c:accessor', NS)
    data = [float(x) for x in arr.text.split()]
    stride = int(acc.get('stride', 3))
    return [data[i:i + stride] for i in range(0, len(data), stride)]


def mat_colors(root):
    """material id -> diffuse rgba, following material -> effect."""
    out = {}
    for m in root.findall('.//c:library_materials/c:material', NS):
        eff = m.find('c:instance_effect', NS).get('url').lstrip('#')
        e = root.find(f".//c:effect[@id='{eff}']", NS)
        d = e.find(".//c:diffuse/c:color", NS) if e is not None else None
        out[m.get('id')] = [float(x) for x in d.text.split()] if d is not None else None
    return out


def primitives(root):
    """[(material, positions, normals, [(vi, ni), ...] per triangle corner)]"""
    out = []
    for node in root.findall('.//c:library_visual_scenes//c:node', NS):
        inst = node.find('c:instance_geometry', NS)
        if inst is None:
            continue
        geom = root.find(f".//c:geometry[@id='{inst.get('url').lstrip('#')}']", NS)
        mesh = geom.find('c:mesh', NS)
        for prim in mesh:
            if tag(prim) not in ('triangles', 'polylist'):
                continue
            inputs = {i.get('semantic'): (int(i.get('offset')), i.get('source'))
                      for i in prim.findall('c:input', NS)}
            nidx = max(o for o, _ in inputs.values()) + 1
            p = [int(x) for x in prim.find('c:p', NS).text.split()]
            corners = [tuple(p[i:i + nidx]) for i in range(0, len(p), nidx)]
            if tag(prim) == 'polylist':
                counts = [int(x) for x in prim.find('c:vcount', NS).text.split()]
            else:
                counts = [3] * (len(corners) // 3)
            tris, at = [], 0
            for n in counts:
                face = corners[at:at + n]
                at += n
                for k in range(1, n - 1):          # fan-triangulate
                    tris.append((face[0], face[k], face[k + 1]))
            vo, vsrc = inputs['VERTEX']
            no, nsrc = inputs.get('NORMAL', (None, None))
            out.append({
                'material': prim.get('material'),
                'pos': floats(root, vsrc),
                'nrm': floats(root, nsrc) if nsrc else None,
                'tris': [[(c[vo], c[no] if no is not None else None) for c in t] for t in tris],
            })
    return out


def write_obj(path, prims, name):
    """Merge the prims sharing one material into a single OBJ."""
    v, vn, faces = [], [], []
    for pr in prims:
        vbase, nbase = len(v), len(vn)
        v.extend(pr['pos'])
        if pr['nrm']:
            vn.extend(pr['nrm'])
        for t in pr['tris']:
            faces.append([(vbase + i + 1, (nbase + n + 1) if (n is not None and pr['nrm']) else None)
                          for i, n in t])
    with open(path, 'w') as f:
        f.write(f"# {name} - converted from ur_description {name.split('_')[0]}.dae\n")
        for p in v:
            f.write("v %.6f %.6f %.6f\n" % tuple(p[:3]))
        for n in vn:
            f.write("vn %.6f %.6f %.6f\n" % tuple(n[:3]))
        for face in faces:
            f.write("f " + " ".join(f"{i}//{n}" if n else str(i) for i, n in face) + "\n")
    return len(v), len(faces)


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for link in LINKS:
        root = ET.parse(src / f"{link}.dae").getroot()
        colors = mat_colors(root)
        prims = primitives(root)
        order, grouped = [], {}
        for pr in prims:
            if pr['material'] not in grouped:
                grouped[pr['material']] = []
                order.append(pr['material'])
            grouped[pr['material']].append(pr)
        parts = []
        for i, m in enumerate(order):
            stem = link if len(order) == 1 else f"{link}_{i}"
            nv, nf = write_obj(dst / f"{stem}.obj", grouped[m], stem)
            parts.append({'mesh': stem, 'dae_material': m,
                          'diffuse': colors.get(m), 'verts': nv, 'tris': nf})
        manifest[link] = parts
        print(f"{link}: " + ", ".join(
            f"{p['mesh']} <- {p['dae_material']} rgb={['%.3f' % c for c in (p['diffuse'] or [])[:3]]} "
            f"({p['tris']} tris)" for p in parts))
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
