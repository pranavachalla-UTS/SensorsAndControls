from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import xml.etree.ElementTree as ET
import xacro

project = Path(__file__).resolve().parent
robots = project / "Robots"

# Work on temporary copies so the downloaded files stay unchanged.
with TemporaryDirectory() as temporary:
    temporary = Path(temporary)

    for folder in ("urdf", "config"):
        shutil.copytree(robots / folder, temporary / folder)

    # Replace ROS package lookups with a local folder path.
    for file in (temporary / "urdf").rglob("*.xacro"):
        text = file.read_text()
        text = text.replace(
            "$(find ur_description)", temporary.as_posix()
        )
        file.write_text(text)

    # Select the UR3 and expand its template.
    description = xacro.process_file(
        str(temporary / "urdf" / "ur.urdf.xacro"),
        mappings={"name": "ur3", "ur_type": "ur3"},
    )

robot = ET.fromstring(description.toxml())

# Point the mesh references to our downloaded robot assets.
for mesh in robot.iter("mesh"):
    filename = mesh.get("filename")
    mesh.set(
        "filename",
        filename.replace(
            "package://ur_description/", "Robots/"
        ),
    )

output = project / "ur3.urdf"
ET.indent(robot)
ET.ElementTree(robot).write(
    output, encoding="utf-8", xml_declaration=True
)

moving_joints = [
    joint for joint in robot.findall("joint")
    if joint.get("type") != "fixed"
]

print("Created:", output.name)
print("Moving joints:", len(moving_joints))
