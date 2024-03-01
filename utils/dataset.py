import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

"""
This file handles the loading and handling of the
cd various datasets needed for training and testing #
"""

def denorm(x):
    """
    Denormalizes x from the range [-1, 1] to the range [0, 1].
    Assumes that the input tensor or value `x` is between [-1, 1].
    Args:
        x (torch.Tensor or float): The tensor or value to be denormalized.
    Returns:
        torch.Tensor or float: The denormalized tensor or value, clamped between 0 and 1.
    """
    out = (x + 1) / 2
    return out.clamp(0, 1)


def load_mnist(file_location='./datasets', image_size=None):
    """
    Loads the MNIST dataset and applies optional transformations. For example, use to load train and test data sets.
    Uses the function torchvision.dataset.MNIST to load the dataset.
    If the dataset is not in the specified folder, it will automatically download it from the internet and store it in
    that folder. Else, it will load from that folder.
        Args:
        file_location (str): The path to the directory where the dataset will be stored. Defaults to './datasets'.

        image_size (tuple or None): The desired size of the images in the dataset.
            If provided, the images will be resized to this size using bilinear interpolation.
            The size should be specified as a tuple (height, width). Defaults to None.

        Returns:
            torch.utils.data.Dataset: The MNIST dataset object.
        """
    if not image_size is None:
        transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.MNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train


def select_from_dataset(dataset, per_class_size, labels):
    """
    Selects a subset of data from a dataset based on specified labels and the desired size per class.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to select data from.
        per_class_size (int): The desired number of samples per class in the subset.
        labels (list): A list of labels to include in the subset.

    Returns:
        torch.utils.data.Subset: The subset of the dataset containing the selected samples.
    """
    # Create a list `indices_by_label` to store the indices of samples for each class in the dataset.
    indices_by_label = [[] for _ in range(10)]

    # Iterate through the dataset and populate the list with the indices of samples based on their labels.
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        indices_by_label[current_class].append(i)
    # Stores the indices of the desired labels in `indices_of_desired_labels`.
    indices_of_desired_labels = [indices_by_label[i] for i in labels]

    # Create a `Subset` object from the original dataset using the selected indices,
    # limited to `per_class_size` number of samples per class
    # Returns the resulting subset of the dataset
    return Subset(dataset, [item for sublist in indices_of_desired_labels for item in sublist[:per_class_size]])


def load_fmnist(file_location='./datasets', image_size=None):
    """
    Loads the FashionMNIST dataset and applies optional transformations. If the dataset is not in the specified folder,
    it will automatically download it from the internet and store it in that folder.
    Else, it will load from that folder.

    Args:
        file_location (str): The path to the directory where the dataset will be stored. Defaults to './datasets'.

        image_size (tuple or None): The desired size of the images in the dataset.
            If provided, the images will be resized to this size using bilinear interpolation.
            The size should be specified as a tuple (height, width). Defaults to None.

    Returns:
        torch.utils.data.Dataset: The FashionMNIST dataset object.
    """
    if not image_size is None:
        transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.FashionMNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train


def load_celeba(file_location='./datasets', image_size=None):
    """
    Loads the CelebA dataset and applies optional transformations.
    The dataset is assumed to be already downloaded and stored in the specified `file_location` directory.

    Args:
        file_location (str): The path to the directory where the dataset will be stored.
            Defaults to './datasets'.

        image_size (tuple or None): The desired size of the images in the dataset.
            If provided, the images will be resized to this size using bilinear interpolation.
            The size should be specified as a tuple (height, width). Defaults to None.

    Returns:
        torch.utils.data.Dataset: The CelebA dataset object.
    """
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Grayscale(),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
    celeba_train = torchvision.datasets.CelebA(root=file_location, target_type="identity", download=False,
                                               transform=transform)
    return celeba_train


def select_from_celeba(dataset, size):
    """
    Selects a subset of data from a CelebA dataset based on the specified size.

    Args:
        dataset (torch.utils.data.Dataset): The CelebA dataset to select data from.

        size (int): The desired size of the subset.

    Returns:
        torch.utils.data.Subset: The subset of the CelebA dataset containing the selected samples.
    """
    return Subset(dataset, range(size))
