import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (InfoNCE Loss) as described in the SimCLR paper.

    Args:
        batch_size (int): Number of samples in a single batch.
        temperature (float): Temperature scaling parameter for the similarity.
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        # Create and cache the mask to ignore self-similarities
        self.mask = (~torch.eye(batch_size * 2, dtype=torch.bool)).float()

    def _move_to_device(self, tensor, reference_tensor):
        """
        Move tensor to the same device as the reference tensor.
        """
        return tensor.to(reference_tensor.device)

    def forward(self, view_1, view_2):
        """
        Compute the contrastive loss for two augmented views of a batch.

        Args:
            view_1 (torch.Tensor): Embeddings from the first view (B x D).
            view_2 (torch.Tensor): Embeddings from the second view (B x D).

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        batch_size = view_1.size(0)

        # Ensure the mask is on the correct device
        mask = self._move_to_device(self.mask, view_1)

        # Normalize embeddings to unit vectors
        normalized_view_1 = F.normalize(view_1, p=2, dim=1)
        normalized_view_2 = F.normalize(view_2, p=2, dim=1)

        # Concatenate the two views and compute the similarity matrix
        embeddings = torch.cat([normalized_view_1, normalized_view_2], dim=0)  # (2B x D)
        similarity_matrix = torch.mm(embeddings, embeddings.T)  # (2B x 2B)

        # Mask out self-similarities (diagonal elements)
        similarity_matrix = similarity_matrix.masked_fill(~mask.bool(), float('-inf'))

        # Extract positive pairs from the similarity matrix
        positives_view_1 = torch.diag(similarity_matrix, batch_size)  # View 1 -> View 2
        positives_view_2 = torch.diag(similarity_matrix, -batch_size)  # View 2 -> View 1
        positives = torch.cat([positives_view_1, positives_view_2], dim=0)  # (2B)

        # Compute the numerator and denominator for the NT-Xent loss
        numerator = torch.exp(positives / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1)

        # Compute the loss
        loss_per_sample = -torch.log(numerator / denominator)
        loss = torch.mean(loss_per_sample)  # Average over all samples

        return loss
