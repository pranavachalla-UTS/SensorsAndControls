from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer

project = Path(__file__).resolve().parents[2]
models = project / "archive" / "models"

# Read the UR3 description we generated.
robot = ET.parse(models / "ur3.urdf").getroot()

# Tell MuJoCo to use the supplied STL collision meshes.
settings = ET.SubElement(robot, "mujoco")
ET.SubElement(
    settings,
    "compiler",
    discardvisual="true",
    strippath="false",
    fusestatic="false",
)

# Save a MuJoCo-compatible URDF.
import_file = models / "ur3_mujoco.urdf"
ET.ElementTree(robot).write(
    import_file, encoding="utf-8", xml_declaration=True
)

# Import it and save MuJoCo's native XML format.
model = mujoco.MjModel.from_xml_path(str(import_file))
mujoco.mj_saveLastXML(str(models / "ur3.xml"), model)

print("Imported UR3 joints:", model.nv)

# Disable gravity ONLY for this preview: no controller exists yet.
model.opt.gravity[:] = 0
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
