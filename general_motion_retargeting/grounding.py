import numpy as np
import mujoco as mj

from .params import ROBOT_VIEWER_XML_DICT


class RobotGrounder:
    def __init__(self, robot_type, geom_groups=(3,)):
        self.model = mj.MjModel.from_xml_path(str(ROBOT_VIEWER_XML_DICT[robot_type]))
        self.data = mj.MjData(self.model)
        self.geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if self.model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_PLANE
            and int(self.model.geom_group[geom_id]) in geom_groups
        ]
        if not self.geom_ids:
            self.geom_ids = [
                geom_id
                for geom_id in range(self.model.ngeom)
                if self.model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_PLANE
            ]

    def _geom_min_z(self, geom_id):
        geom_type = self.model.geom_type[geom_id]
        geom_pos = self.data.geom_xpos[geom_id]
        geom_mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self.model.geom_size[geom_id]

        if geom_type == mj.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[geom_id]
            start = self.model.mesh_vertadr[mesh_id]
            count = self.model.mesh_vertnum[mesh_id]
            verts = self.model.mesh_vert[start : start + count] * self.model.mesh_scale[mesh_id]
            return float((geom_pos + verts @ geom_mat.T)[:, 2].min())

        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = geom_size[:3]
            corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
            return float((geom_pos + corners @ geom_mat.T)[:, 2].min())

        if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            return float(geom_pos[2] - geom_size[0])

        if geom_type in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            axis = geom_mat[:, 2]
            radius = geom_size[0]
            half_length = geom_size[1]
            z_extent = abs(axis[2]) * half_length + radius * np.sqrt(max(0.0, 1.0 - axis[2] ** 2))
            if geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
                z_extent += radius * abs(axis[2])
            return float(geom_pos[2] - z_extent)

        return float(geom_pos[2])

    def qpos_min_z(self, qpos):
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        return min(self._geom_min_z(geom_id) for geom_id in self.geom_ids)

    def ground_qpos(self, qpos, ground_offset=0.0):
        grounded = np.array(qpos, copy=True)
        grounded[2] -= self.qpos_min_z(grounded) - ground_offset
        return grounded

    def ground_qpos_sequence(self, qpos, ground_offset=0.0, per_frame=False):
        grounded = np.array(qpos, copy=True)
        min_z = np.array([self.qpos_min_z(frame_qpos) for frame_qpos in grounded])
        if per_frame:
            grounded[:, 2] -= min_z - ground_offset
        else:
            grounded[:, 2] -= float(min_z.min()) - ground_offset
        return grounded
