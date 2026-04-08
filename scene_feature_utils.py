import numpy as np


class SceneFeatureBank:
    def __init__(
        self,
        centers_xy,
        heights,
        footprint_areas,
        volumes,
        permittivities,
        conductivities,
        scattering_coefficients,
        scene_diag_xy,
        use_material_features=False,
    ):
        self.centers_xy = centers_xy.astype(np.float32)
        self.heights = heights.astype(np.float32)
        self.footprint_areas = footprint_areas.astype(np.float32)
        self.volumes = volumes.astype(np.float32)
        self.permittivities = permittivities.astype(np.float32)
        self.conductivities = conductivities.astype(np.float32)
        self.scattering_coefficients = scattering_coefficients.astype(np.float32)
        self.scene_diag_xy = float(scene_diag_xy)
        self.use_material_features = bool(use_material_features)

    @classmethod
    def from_scene(cls, scene, materials=None, use_material_features=False):
        centers_xy = []
        heights = []
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
            centers_xy.append(center_xy)
            height = float(getattr(obj, "height", obj_bbox[1, 2] - obj_bbox[0, 2]))
            footprint_area = float(getattr(obj, "footprint_area", 0.0))
            try:
                volume = float(obj.volume)
            except Exception:
                bbox_dims = obj_bbox[1] - obj_bbox[0]
                volume = float(max(bbox_dims[0], 0.0) * max(bbox_dims[1], 0.0) * max(bbox_dims[2], 0.0))

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

            heights.append(height)
            footprint_areas.append(footprint_area)
            volumes.append(volume)
            permittivities.append(permittivity)
            conductivities.append(conductivity)
            scattering_coefficients.append(scattering_coefficient)

        if not centers_xy:
            centers_xy = np.zeros((0, 2), dtype=np.float32)
            heights = np.zeros((0,), dtype=np.float32)
            footprint_areas = np.zeros((0,), dtype=np.float32)
            volumes = np.zeros((0,), dtype=np.float32)
            permittivities = np.zeros((0,), dtype=np.float32)
            conductivities = np.zeros((0,), dtype=np.float32)
            scattering_coefficients = np.zeros((0,), dtype=np.float32)
        else:
            centers_xy = np.stack(centers_xy, axis=0)
            heights = np.asarray(heights, dtype=np.float32)
            footprint_areas = np.asarray(footprint_areas, dtype=np.float32)
            volumes = np.asarray(volumes, dtype=np.float32)
            permittivities = np.asarray(permittivities, dtype=np.float32)
            conductivities = np.asarray(conductivities, dtype=np.float32)
            scattering_coefficients = np.asarray(scattering_coefficients, dtype=np.float32)

        return cls(
            centers_xy,
            heights,
            footprint_areas,
            volumes,
            permittivities,
            conductivities,
            scattering_coefficients,
            scene_diag_xy,
            use_material_features=use_material_features,
        )

    @classmethod
    def from_dataset(cls, dataset, use_material_features=False):
        materials = dataset["materials"] if use_material_features and "materials" in dataset else None
        return cls.from_scene(dataset["scene"], materials=materials, use_material_features=use_material_features)

    def _sorted_indices_from_point(self, point_xy):
        if self.centers_xy.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        dists = np.linalg.norm(self.centers_xy - point_xy[None, :], axis=1)
        order = np.argsort(dists)
        return order.astype(np.int64), dists[order].astype(np.float32)

    def local_features(self, point_xy, nearest_k=5, radii=(25.0, 50.0, 100.0)):
        point_xy = np.asarray(point_xy, dtype=np.float32)
        order, sorted_dists = self._sorted_indices_from_point(point_xy)

        nearest_dists = np.full((nearest_k,), self.scene_diag_xy, dtype=np.float32)
        nearest_heights = np.zeros((nearest_k,), dtype=np.float32)
        nearest_areas = np.zeros((nearest_k,), dtype=np.float32)
        nearest_permittivities = np.zeros((nearest_k,), dtype=np.float32)
        nearest_conductivities = np.zeros((nearest_k,), dtype=np.float32)
        nearest_scattering = np.zeros((nearest_k,), dtype=np.float32)

        use_k = min(nearest_k, len(order))
        if use_k > 0:
            nearest_idx = order[:use_k]
            nearest_dists[:use_k] = sorted_dists[:use_k]
            nearest_heights[:use_k] = self.heights[nearest_idx]
            nearest_areas[:use_k] = self.footprint_areas[nearest_idx]
            if self.use_material_features:
                nearest_permittivities[:use_k] = self.permittivities[nearest_idx]
                nearest_conductivities[:use_k] = self.conductivities[nearest_idx]
                nearest_scattering[:use_k] = self.scattering_coefficients[nearest_idx]

        count_feats = []
        max_height_feats = []
        if len(order) == 0:
            count_feats = [0.0 for _ in radii]
            max_height_feats = [0.0 for _ in radii]
        else:
            all_dists = np.linalg.norm(self.centers_xy - point_xy[None, :], axis=1)
            for radius in radii:
                mask = all_dists <= radius
                count_feats.append(float(mask.sum()))
                max_height_feats.append(float(self.heights[mask].max()) if np.any(mask) else 0.0)

        pieces = [
            nearest_dists,
            nearest_heights,
            nearest_areas,
        ]
        if self.use_material_features:
            pieces.extend([nearest_permittivities, nearest_conductivities, nearest_scattering])
        pieces.extend([
            np.asarray(count_feats, dtype=np.float32),
            np.asarray(max_height_feats, dtype=np.float32),
        ])
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def corridor_features(self, tx_xy, rx_xy, top_k=5, corridor_radii=(25.0, 50.0, 100.0), n_bins=8):
        tx_xy = np.asarray(tx_xy, dtype=np.float32)
        rx_xy = np.asarray(rx_xy, dtype=np.float32)

        empty_dim = top_k * (7 if self.use_material_features else 4) + len(corridor_radii) + n_bins * 2
        if self.centers_xy.shape[0] == 0:
            return np.zeros((empty_dim,), dtype=np.float32)

        seg = rx_xy - tx_xy
        seg_norm_sq = float(np.dot(seg, seg))
        rel = self.centers_xy - tx_xy[None, :]

        if seg_norm_sq < 1e-8:
            progress = np.zeros((self.centers_xy.shape[0],), dtype=np.float32)
            closest = np.repeat(tx_xy[None, :], self.centers_xy.shape[0], axis=0)
        else:
            progress = np.clip((rel @ seg) / seg_norm_sq, 0.0, 1.0).astype(np.float32)
            closest = tx_xy[None, :] + progress[:, None] * seg[None, :]

        dist_to_segment = np.linalg.norm(self.centers_xy - closest, axis=1).astype(np.float32)
        order = np.argsort(dist_to_segment)

        top_dists = np.full((top_k,), self.scene_diag_xy, dtype=np.float32)
        top_progress = np.zeros((top_k,), dtype=np.float32)
        top_heights = np.zeros((top_k,), dtype=np.float32)
        top_areas = np.zeros((top_k,), dtype=np.float32)
        top_permittivities = np.zeros((top_k,), dtype=np.float32)
        top_conductivities = np.zeros((top_k,), dtype=np.float32)
        top_scattering = np.zeros((top_k,), dtype=np.float32)

        use_k = min(top_k, len(order))
        if use_k > 0:
            top_idx = order[:use_k]
            top_dists[:use_k] = dist_to_segment[top_idx]
            top_progress[:use_k] = progress[top_idx]
            top_heights[:use_k] = self.heights[top_idx]
            top_areas[:use_k] = self.footprint_areas[top_idx]
            if self.use_material_features:
                top_permittivities[:use_k] = self.permittivities[top_idx]
                top_conductivities[:use_k] = self.conductivities[top_idx]
                top_scattering[:use_k] = self.scattering_coefficients[top_idx]

        corridor_counts = []
        for radius in corridor_radii:
            corridor_counts.append(float((dist_to_segment <= radius).sum()))

        bin_counts = np.zeros((n_bins,), dtype=np.float32)
        bin_max_heights = np.zeros((n_bins,), dtype=np.float32)
        main_radius = float(max(corridor_radii))
        valid_mask = dist_to_segment <= main_radius
        if np.any(valid_mask):
            valid_progress = progress[valid_mask]
            valid_heights = self.heights[valid_mask]
            bin_ids = np.clip((valid_progress * n_bins).astype(np.int64), 0, n_bins - 1)
            for bin_idx in range(n_bins):
                mask = bin_ids == bin_idx
                if np.any(mask):
                    bin_counts[bin_idx] = float(mask.sum())
                    bin_max_heights[bin_idx] = float(valid_heights[mask].max())

        pieces = [top_dists, top_progress, top_heights, top_areas]
        if self.use_material_features:
            pieces.extend([top_permittivities, top_conductivities, top_scattering])
        pieces.extend([
            np.asarray(corridor_counts, dtype=np.float32),
            bin_counts,
            bin_max_heights,
        ])
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def build_feature_vector(self, tx_pos, rx_pos, nearest_k=5, corridor_k=5, radii=(25.0, 50.0, 100.0), corridor_bins=8):
        tx_pos = np.asarray(tx_pos, dtype=np.float32)
        rx_pos = np.asarray(rx_pos, dtype=np.float32)
        tx_xy = tx_pos[:2]
        rx_xy = rx_pos[:2]

        rx_local = self.local_features(rx_xy, nearest_k=nearest_k, radii=radii)
        tx_local = self.local_features(tx_xy, nearest_k=nearest_k, radii=radii)
        corridor = self.corridor_features(tx_xy, rx_xy, top_k=corridor_k, corridor_radii=radii, n_bins=corridor_bins)
        return np.concatenate([rx_local, tx_local, corridor], axis=0).astype(np.float32)
