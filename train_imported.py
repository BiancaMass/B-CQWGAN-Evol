import os
import argparse
import math
import numpy as np
import torch
import csv
# import torch_directml #directml does not support complex data types
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

from utils.dataset import load_mnist, load_fmnist, denorm, select_from_dataset
from utils.wgan import compute_gradient_penalty
from models.QGCC_imported import PQWGAN_CC_imported


def train_imported(classes_str: str, dataset_str: str, out_folder: str, randn: bool,
                   qasm_file_path:str, metadata_file_path, batch_size: int = 25, n_epochs: int = 50,
                   image_size: int = 28):
    """
    Trains the generator and discriminator of the PQWGAN, using an imported circuit structure.

    :param classes_str: str. Classes of images to generate, e.g., '01' for MNIST.
    :param dataset_str: str. The name of the dataset to train on, e.g., 'mnist'.
    :param metadata_file_path: str. File with metadata for the circuit.
    :param batch_size: int. The size of each batch of data.
    :param n_epochs: int. The number of epochs to train the model.
    :param out_folder: str. The directory where the training outputs will be saved.
    :param randn: bool. Whether to draw the latent vector from uniform or normal distribution.
    :param qasm_file_path: str. file path to the QASM file to import the circuit for the generator.
    :return: None.
    """
    with open(metadata_file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            variable_name = row['Variable'].lower().replace(' ', '_')

            if variable_name == 'n_data_qubits':
                n_data_qubits = int(row['Value'])
            elif variable_name == 'n_ancilla':
                n_ancillas = int(row['Value'])
            elif variable_name == 'image_shape':
                patch_shape = eval(row['Value'])  # Converts string to tuple

    classes = list(set([int(digit) for digit in classes_str]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    n_epochs = n_epochs
    image_size = image_size
    channels = 1
    if dataset_str == "mnist":
        dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    elif dataset_str == "fmnist":
        dataset = select_from_dataset(load_fmnist(image_size=image_size), 1000, classes)
    else:
        raise ValueError("Unsupported dataset")

    ancillas = n_ancillas
    n_data_qubits = n_data_qubits
    qubits = n_data_qubits + ancillas

    n_sub_generators = int(image_size/(int(patch_shape[0])))

    # Default training parameters
    lr_D = 0.0002
    lr_G = 0.01
    b1 = 0
    b2 = 0.9
    latent_dim = qubits  # length of latent vector = number of qubits (incl.  ancilla(s))
    lambda_gp = 10
    # How often to train gen and critic.E.g., if n_critic=5, train the gen every 5 critics.
    n_critic = 5
    sample_interval = 10
    # Default output folder name. Change if you want to include more params.
    out_dir = f"{out_folder}/{classes_str}_{batch_size}bs"
    if randn:
        out_dir += "_randn"

    os.makedirs(out_dir, exist_ok=False)  # if dir already exists, raises an error

    gan = PQWGAN_CC_imported(image_size=image_size,
                             channels=channels,
                             n_sub_generators=n_sub_generators,
                             n_ancillas=ancillas,
                             qasm_file_path=qasm_file_path)

    # Assign the critic (discr.) and generator models to the target device (e.g., GPU, CPU).
    critic = gan.critic.to(device)
    generator = gan.generator.to(device)

    # DataLoader from Pytorch to efficiently load and iterate over batches from the given dataset.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # Initialize an Adam optimizer for the generator.
    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    # Initialize an Adam optimizer for the critic.
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))

    # Generate latent vectors.
    # If randn is True, use normally distributed random numbers
    # Else, use uniform random numbers.
    if randn:
        fixed_z = torch.randn(batch_size, latent_dim, device=device)
    else:
        fixed_z = torch.rand(batch_size, latent_dim, device=device)

    wasserstein_distance_history = []  # Store the Wasserstein distances
    saved_initial = False
    batches_done = 0

    # Begin training process of Generator and Discriminator.
    for epoch in range(n_epochs):
        print(f'Epoch number {epoch} \n')
        # Iterate over batches in the data loader.
        for i, (real_images, _) in enumerate(dataloader):
            if not saved_initial:
                fixed_images = generator(fixed_z)
                save_image(denorm(fixed_images),
                           os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(real_images), os.path.join(out_dir, 'real_samples.png'), nrow=5)
                saved_initial = True

            # Move real images to the specified device.
            real_images = real_images.to(device)
            # Initialize the critic's optimizer (pytorch zero_grad).
            optimizer_C.zero_grad()

            if randn:  # latent vector from normal distribution
                z = torch.randn(batch_size, latent_dim, device=device)
            else:  # latent vector from uniform distribution
                z = torch.rand(batch_size, latent_dim, device=device)

            # Give generator latent vector z to generate images.
            fake_images = generator(z)

            # Compute the critic's predictions for real and fake images.
            real_validity = critic(real_images)  # Real images.
            fake_validity = critic(fake_images)  # Fake images.
            # Calculate the gradient penalty and adversarial loss.
            gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
            d_loss = -torch.mean(real_validity) + torch.mean(
                fake_validity) + lambda_gp * gradient_penalty
            # Calculate Wasserstein distance.
            wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
            # Add distance for this batch.
            wasserstein_distance_history.append(wasserstein_distance.item())

            # Backpropagate and update the critic's weights.
            d_loss.backward()
            optimizer_C.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_images = generator(z)
                # Loss measures generator's ability to fool the discriminator
                fake_validity = critic(fake_images)
                g_loss = -torch.mean(fake_validity)

                # Backpropagate and update the generator's weights
                g_loss.backward()
                optimizer_G.step()

                # Print and log the training progress
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Wasserstein Distance: {wasserstein_distance.item()}]")
                # Save Wasserstein distance history to a file
                np.save(os.path.join(out_dir, 'wasserstein_distance.npy'),
                        wasserstein_distance_history)
                # Update the total number of batches done
                batches_done += n_critic

                # Save generated images and model states at regular intervals
                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z)
                    save_image(denorm(fixed_images),
                               os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                    torch.save(critic.state_dict(),
                               os.path.join(out_dir, 'critic-{}.pt'.format(batches_done)))
                    torch.save(generator.state_dict(),
                               os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
                    print("saved images and state")


# Define the command-line arguments using argparse
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PQWGAN with an imported quantum circuit.")
    parser.add_argument("--classes", "-cl", type=str, help="Classes to train on, e.g., '0123456789' for MNIST.")
    parser.add_argument("--classes", "-cl", type=str, help="Classes to train on, e.g., '0123456789' for MNIST.")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset to train on, either 'mnist' or 'fmnist'.")
    parser.add_argument("--batch_size", "-b", type=int, help="Batch size for training.")
    parser.add_argument("--out_folder", "-o", type=str, help="Output directory for saving models and images.")
    parser.add_argument("--randn", "-rn", action="store_true", help="Use normal distribution for generating latent vectors; otherwise, use uniform distribution.")
    parser.add_argument("--qasm_file_path", "-qf", type=str, help="Path to the QASM file for the quantum circuit.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Adjust the call to the training function according to the updated argument list
    train_imported(classes_str=args.classes,
                   dataset_str=args.dataset,
                   batch_size=args.batch_size,
                   out_folder=args.out_folder,
                   randn=args.randn,
                   qasm_file_path=args.qasm_file_path)

