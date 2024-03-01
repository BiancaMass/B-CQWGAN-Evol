import os
import torch.distributed as dist

# This file defines setup() and cleanup() used for setting up and cleaning up
# the distributed environment i.e. for PARALLEL COMPUTATION.


def setup(rank, world_size):
    """
    Sets up the distributed environment for training.
    Sets the environment variables 'MASTER_ADDR' and 'MASTER_PORT' to 'localhost' and '12382', respectively.
    Initializes the process group using the 'gloo' backend and the provided 'rank' and 'world_size'.
    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Note: When running the code on a local machine, the distributed computation will be simulated using
    multiple processes within that machine. Each process will have a unique rank assigned to it,
    allowing them to communicate and coordinate as if they were running on separate machines.
    It's important to note that while the code can be executed on a local machine, the benefits of
    distributed computing, such as parallel processing and improved performance, may not be fully realized.
    """
    print(f"Running setup on rank {rank}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12382'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment after training.
    Destroys the process group, freeing up resources."""
    dist.destroy_process_group()


# Distributed computation is the execution of a computational task across multiple interconnected devices
# allowing for parallel processing.
#
# A distributed environment is a setup that enables the execution of distributed computations
# by providing the necessary infrastructure, communication protocols, and coordination mechanisms.

# Setting up the distributed environment typically involves performing necessary configurations
# and initialization steps to establish communication channels between the different processes
# or devices participating in the distributed computation

# Cleaning up the distributed environment involves releasing the resources and connections
# used during the distributed computation. This includes destroying the process group
# and freeing up any allocated resources, such as network connections or shared memory.
