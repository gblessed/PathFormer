import numpy as np


class SceneObjectTokenBank:
    def __init__(
        self,
        centers_xy,
        dims_xyz,
        footprint_areas,
        volumes,
        permittivities,
        conductivities,
        scattering_coefficients,
        scene_diag_xy,
        use_material_features=False,
    ):
        self.centers_xy = centers_xy.astype(np.float32)
        self.dims_xyz = dims_xyz.astype(np.float32)
        self.footprint_areas = footprint_areas.astype(np.float32)
        self.volumes = volumes.astype(np.float32)
        self.permittivities = permittivities.astype(np.float32)
        self.conductivities = conductivities.astype(np.float32)
        self.scattering_coefficients = scattering_coefficients.astype(np.float32)
        self.scene_diag_xy = float(max(scene_diag_xy, 1e-6))
        self.use_material_features = bool(use_material_features)

    @classmethod
    def from_scene(cls, scene, materials=None, use_material_features=False):
        centers_xy = []
        dims_xyz = []
        footprint_areas = []
        volumes = []
        permittivities = []
        conductivities = []
        scattering_coefficients = []

        bbox = np.array(scene.bounding_box.bounds, dtype=np.float32)
        scene_diag_xy = np.linalg.norm(bbox[1, :2] - bbox[0, :2])

        for obj in scene.objects:
            if getattr(obj, "label", None) != "buildings":
                continue
            obj_bbox = np.array(obj.bounding_box.bounds, dtype=np.float32)
            center_xy = 0.5 * (obj_bbox[0, :2] + obj_bbox[1, :2])
            dims = np.maximum(obj_bbox[1] - obj_bbox[0], 0.0)
            footprint_area = float(getattr(obj, "footprint_area", dims[0] * dims[1]))
            try:
                volume = float(obj.volume)
            except Exception:
                volume = float(dims[0] * dims[1] * dims[2])

            permittivity = 0.0
            conductivity = 0.0
            scattering_coefficient = 0.0
            if use_material_features and materials is not None:
                material_ids = getattr(obj, "materials", None)
                material_id = None
                if isinstance(material_ids, (list, tuple, np.ndarray)) and len(material_ids) > 0:
                    material_id = int(material_ids[0])
                elif material_ids is not None:
                    material_id = int(material_ids)
                if material_id is not None and 0 <= material_id < len(materials):
                    material = materials[material_id]
                    permittivity = float(getattr(material, "permittivity", 0.0))
                    conductivity = float(getattr(material, "conductivity", 0.0))
                    scattering_coefficient = float(getattr(material, "scattering_coefficient", 0.0))

            centers_xy.append(center_xy)
            dims_xyz.append(dims)
            footprint_areas.append(footprint_area)
            volumes.append(volume)
            permittivities.append(permittivity)
            conductivities.append(conductivity)
            scattering_coefficients.append(scattering_coefficient)

        if not centers_xy:
            centers_xy = np.zeros((0, 2), dtype=np.float32)
            dims_xyz = np.zeros((0, 3), dtype=np.float32)
            footprint_areas = np.zeros((0,), dtype=np.float32)
            volumes = np.zeros((0,), dtype=np.float32)
            permittivities = np.zeros((0,), dtype=np.float32)
            conductivities = np.zeros((0,), dtype=np.float32)
            scattering_coefficients = np.zeros((0,), dtype=np.float32)
        else:
            centers_xy = np.stack(centers_xy, axis=0)
            dims_xyz = np.stack(dims_xyz, axis=0)
            footprint_areas = np.asarray(footprint_areas, dtype=np.float32)
            volumes = np.asarray(volumes, dtype=np.float32)
            permittivities = np.asarray(permittivities, dtype=np.float32)
            conductivities = np.asarray(conductivities, dtype=np.float32)
            scattering_coefficients = np.asarray(scattering_coefficients, dtype=np.float32)

        return cls(
            centers_xy=centers_xy,
            dims_xyz=dims_xyz,
            footprint_areas=footprint_areas,
            volumes=volumes,
            permittivities=permittivities,
            conductivities=conductivities,
            scattering_coefficients=scattering_coefficients,
            scene_diag_xy=scene_diag_xy,
            use_material_features=use_material_features,
        )

    @classmethod
    def from_dataset(cls, dataset, use_material_features=False):
        materials = dataset["materials"] if use_material_features and "materials" in dataset else None
        return cls.from_scene(dataset["scene"], materials=materials, use_material_features=use_material_features)

    def feature_dim(self):
        return 18 + (3 if self.use_material_features else 0)

    def _corridor_geometry(self, tx_xy, rx_xy):
        seg = rx_xy - tx_xy
        seg_norm_sq = float(np.dot(seg, seg))
        rel = self.centers_xy - tx_xy[None, :]
        if seg_norm_sq < 1e-8:
            progress = np.zeros((self.centers_xy.shape[0],), dtype=np.float32)
            closest = np.repeat(tx_xy[None, :], self.centers_xy.shape[0], axis=0)
        else:
            progress = np.clip((rel @ seg) / seg_norm_sq, 0.0, 1.0).astype(np.float32)
            closest = tx_xy[None, :] + progress[:, None] * seg[None, :]
        corridor_vec = self.centers_xy - closest
        dist_corr = np.linalg.norm(corridor_vec, axis=1).astype(np.float32)
        if np.linalg.norm(seg) < 1e-8:
            signed_lat = np.zeros_like(dist_corr)
        else:
            seg_unit = seg / (np.linalg.norm(seg) + 1e-8)
            signed_lat = rel[:, 0] * seg_unit[1] - rel[:, 1] * seg_unit[0]
            signed_lat = signed_lat.astype(np.float32)
        return progress, dist_corr, signed_lat

    def _select_indices(self, tx_xy, rx_xy, max_objects, nearest_rx_k, nearest_tx_k, corridor_k):
        n = self.centers_xy.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.int64)

        dist_rx = np.linalg.norm(self.centers_xy - rx_xy[None, :], axis=1)
        dist_tx = np.linalg.norm(self.centers_xy - tx_xy[None, :], axis=1)
        progress, dist_corr, _ = self._corridor_geometry(tx_xy, rx_xy)

        candidates = []
        candidates.extend(np.argsort(dist_rx)[: min(nearest_rx_k, n)].tolist())
        candidates.extend(np.argsort(dist_tx)[: min(nearest_tx_k, n)].tolist())
        candidates.extend(np.argsort(dist_corr)[: min(corridor_k, n)].tolist())

        seen = set()
        deduped = []
        for idx in candidates:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)

        scored = []
        for idx in deduped:
            scored.append((dist_corr[idx], dist_rx[idx], dist_tx[idx], progress[idx], idx))
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        out = [idx for *_, idx in scored[:max_objects]]
        return np.asarray(out, dtype=np.int64)

    def build_object_tokens(
        self,
        tx_pos,
        rx_pos,
        max_objects=24,
        nearest_rx_k=8,
        nearest_tx_k=4,
        corridor_k=12,
    ):
        tx_pos = np.asarray(tx_pos, dtype=np.float32)
        rx_pos = np.asarray(rx_pos, dtype=np.float32)
        tx_xy = tx_pos[:2]
        rx_xy = rx_pos[:2]

        token_dim = self.feature_dim()
        tokens = np.zeros((max_objects, token_dim), dtype=np.float32)
        padding_mask = np.ones((max_objects,), dtype=bool)

        if self.centers_xy.shape[0] == 0:
            return tokens, padding_mask

        selected = self._select_indices(tx_xy, rx_xy, max_objects, nearest_rx_k, nearest_tx_k, corridor_k)
        if len(selected) == 0:
            return tokens, padding_mask

        progress, dist_corr, signed_lat = self._corridor_geometry(tx_xy, rx_xy)
        centers = self.centers_xy[selected]
        dims = self.dims_xyz[selected]
        dist_tx = np.linalg.norm(centers - tx_xy[None, :], axis=1).astype(np.float32)
        dist_rx = np.linalg.norm(centers - rx_xy[None, :], axis=1).astype(np.float32)
        rel_tx = (centers - tx_xy[None, :]) / self.scene_diag_xy
        rel_rx = (centers - rx_xy[None, :]) / self.scene_diag_xy

        angle_tx = np.arctan2(centers[:, 1] - tx_xy[1], centers[:, 0] - tx_xy[0]).astype(np.float32)
        angle_rx = np.arctan2(centers[:, 1] - rx_xy[1], centers[:, 0] - rx_xy[0]).astype(np.float32)

        feature_pieces = [
            rel_tx[:, 0:1],
            rel_tx[:, 1:2],
            rel_rx[:, 0:1],
            rel_rx[:, 1:2],
            dims / self.scene_diag_xy,
            (self.footprint_areas[selected] / (self.scene_diag_xy ** 2))[:, None],
            (self.volumes[selected] / (self.scene_diag_xy ** 3))[:, None],
            (dist_tx / self.scene_diag_xy)[:, None],
            (dist_rx / self.scene_diag_xy)[:, None],
            (dist_corr[selected] / self.scene_diag_xy)[:, None],
            progress[selected][:, None],
            (signed_lat[selected] / self.scene_diag_xy)[:, None],
            np.cos(angle_tx)[:, None],
            np.sin(angle_tx)[:, None],
            np.cos(angle_rx)[:, None],
            np.sin(angle_rx)[:, None],
        ]
        if self.use_material_features:
            feature_pieces.extend(
                [
                    self.permittivities[selected][:, None],
                    self.conductivities[selected][:, None],
                    self.scattering_coefficients[selected][:, None],
                ]
            )

        token_mat = np.concatenate(feature_pieces, axis=1).astype(np.float32)
        use_n = min(max_objects, token_mat.shape[0])
        tokens[:use_n] = token_mat[:use_n]
        padding_mask[:use_n] = False
        return tokens, padding_mask
