import matplotlib.pyplot as plt
import torch

def plot_matrix(tensor, title="Matrix Plot", cmap="viridis"):
    """
    Plots a PyTorch tensor as a matrix using matplotlib.

    Args:
        tensor (torch.Tensor): The tensor to plot. Should be 2D.
        title (str): Title of the plot. Default is "Matrix Plot".
        cmap (str): Colormap to use for the plot. Default is "viridis".

    Raises:
        ValueError: If the input tensor is not 2D.
    """
    # Check if the tensor is 2D
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D to plot as a matrix.")

    # Convert tensor to numpy array for plotting
    matrix = tensor.cpu().numpy()

    # Create the plot
    plt.figure(figsize=(8, 6), dpi=80)
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.grid(False)  # Disable grid for a cleaner matrix view
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create a sample 2D tensor
    sample_tensor = torch.randn(10, 10)  # 10x10 matrix with random values
    plot_tensor_as_matrix(sample_tensor, title="Sample Tensor Matrix")
