import torch
import MinkowskiEngine as ME


class CollateFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        for d in list_data:
            list_d.append(
                (
                    d["coordinates"].to(self.device),
                    d["features"].to(self.device),
                    d["labels"],
                )
            )
            list_idx.append(d["idx"].view(-1, 1))

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(
            dtype=torch.float32, device=self.device
        )(list_d)
        idx = torch.cat(list_idx, dim=0)
        return {
            "coordinates": coordinates_batch,
            "features": features_batch,
            "labels": labels_batch,
            "idx": idx,
        }


class CollateFNPseudo:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_data_batch = []
        list_pseudo = []

        selected = []
        idx = []

        for d in list_data:
            list_data_batch.append(
                (
                    d["coordinates"].to(self.device),
                    d["features"].to(self.device),
                    d["labels"],
                )
            )
            list_pseudo.append(d["pseudo_labels"].to(self.device))
            selected.append(d["sampled_idx"].unsqueeze(0))
            idx.append(d["idx"].unsqueeze(0))

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(
            dtype=torch.float32, device=self.device
        )(list_data_batch)

        idx = torch.cat(idx, dim=0)
        pseudo_labels = torch.cat(list_pseudo, dim=0)

        return {
            "coordinates": coordinates_batch,
            "features": features_batch,
            "labels": labels_batch,
            "pseudo_labels": pseudo_labels,
            "sampled_idx": selected,
            "idx": idx,
        }


class CollateMerged:
    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device

    def __call__(self, list_data) -> dict:
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """

        source1_list_data = [
            (
                d["source1_coordinates"].to(self.device),
                d["source1_features"].to(self.device),
                d["source1_labels"],
            )
            for d in list_data
        ]

        source2_list_data = [
            (
                d["source2_coordinates"].to(self.device),
                d["source2_features"].to(self.device),
                d["source2_labels"],
            )
            for d in list_data
        ]

        target_list_data = [
            (
                d["target_coordinates"].to(self.device),
                d["target_features"].to(self.device),
                d["target_labels"],
            )
            for d in list_data
        ]

        source1_coordinates_batch, source1_features_batch, source1_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                source1_list_data
            )
        )

        source2_coordinates_batch, source2_features_batch, source2_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                source2_list_data
            )
        )

        target_coordinates_batch, target_features_batch, target_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                target_list_data
            )
        )

        return_dict = {
            "source1_coordinates": source1_coordinates_batch,
            "source1_features": source1_features_batch,
            "source1_labels": source1_labels_batch,
            "source2_coordinates": source2_coordinates_batch,
            "source2_features": source2_features_batch,
            "source2_labels": source2_labels_batch,
            "target_coordinates": target_coordinates_batch,
            "target_features": target_features_batch,
            "target_labels": target_labels_batch,
        }

        return return_dict


class CollateMergedPseudo:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        source1_list_data = []
        source2_list_data = []
        target_list_data = []
        target_list_pseudo = []

        source1_selected = []
        source2_selected = []
        target_selected = []

        source1_idx = []
        source2_idx = []
        target_idx = []

        for d in list_data:
            source1_list_data.append(
                (
                    d["source1_coordinates"].to(self.device),
                    d["source1_features"].to(self.device),
                    d["source1_labels"],
                )
            )

            source2_list_data.append(
                (
                    d["source2_coordinates"].to(self.device),
                    d["source2_features"].to(self.device),
                    d["source2_labels"],
                )
            )

            target_list_data.append(
                (
                    d["target_coordinates"].to(self.device),
                    d["target_features"].to(self.device),
                    d["target_labels"],
                )
            )
            target_list_pseudo.append(
                (
                    d["target_coordinates"].to(self.device),
                    d["target_features"].to(self.device),
                    d["target_pseudo_labels"].to(self.device),
                )
            )

            source1_selected.append(d["source1_sampled_idx"])
            source2_selected.append(d["source1_sampled_idx"])
            target_selected.append(d["target_sampled_idx"])
            source2_idx.append(d["source2_idx"].unsqueeze(0))
            target_idx.append(d["target_idx"].unsqueeze(0))

        source1_coordinates_batch, source1_features_batch, source1_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                source1_list_data
            )
        )

        source2_coordinates_batch, source2_features_batch, source2_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                source2_list_data
            )
        )

        target_coordinates_batch, target_features_batch, target_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                target_list_data
            )
        )

        _, _, target_pseudo_labels = ME.utils.SparseCollation(
            dtype=torch.float32, device=self.device
        )(target_list_pseudo)

        return {
            "source1_coordinates": source1_coordinates_batch,
            "source1_features": source1_features_batch,
            "source1_labels": source1_labels_batch,
            "source2_coordinates": source2_coordinates_batch,
            "source2_features": source2_features_batch,
            "source2_labels": source2_labels_batch,
            "target_coordinates": target_coordinates_batch,
            "target_features": target_features_batch,
            "target_labels": target_labels_batch,
            "target_pseudo_labels": target_pseudo_labels,
            "source1_sampled": source1_selected,
            "source2_sampled": source2_selected,
            "target_sampled": target_selected,
            "source1_idx": source1_idx,
            "source2_idx": source2_idx,
            "target_idx": target_idx,
        }


class CollateMixed:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        list_data has a list of dicts with keys:
            - mixed_coordinates
            - mixed_labels
            - mixed_gt_labels
            - mixed_features
            - separation_idx
            - mixed_sampled_idx
            - mixed_idx
        """
        gt_list_data = []
        pseudo_list_data = []

        separation_list = []
        mixed_sampled = []
        mixed_idx = []

        for d in list_data:
            gt_list_data.append(
                (
                    d["mixed_coordinates"].to(self.device),
                    d["mixed_features"].to(self.device),
                    d["mixed_gt_labels"],
                )
            )
            pseudo_list_data.append(
                (
                    d["mixed_coordinates"].to(self.device),
                    d["mixed_features"].to(self.device),
                    d["mixed_labels"],
                )
            )

            mixed_sampled.append(d["mixed_sampled"])
            separation_list.append(d["separation_idx"].unsqueeze(0))
            mixed_idx.append(d["mixed_idx"].unsqueeze(0))

        _, _, gt_labels_batch = ME.utils.SparseCollation(
            dtype=torch.float32, device=self.device
        )(gt_list_data)

        pseudo_coordinates_batch, pseudo_features_batch, pseudo_labels_batch = (
            ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(
                pseudo_list_data
            )
        )

        separation_list = torch.cat(separation_list, dim=0)
        mixed_idx = torch.cat(mixed_idx, dim=0)

        return {
            "mixed_coordinates": pseudo_coordinates_batch,
            "mixed_features": pseudo_features_batch,
            "mixed_labels": pseudo_labels_batch,
            "mixed_gt_labels": gt_labels_batch,
            "mixed_sampled": mixed_sampled,
            "separation_idx": separation_list,
            "mixed_idx": mixed_idx,
        }


class CollateMixedMasked:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        """
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        list_data has a list of dicts with keys:

        """
        pass
