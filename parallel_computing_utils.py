import os
import torch


def init():
    # Define master address and port
    import idr_torch

    # Get task information
    global_rank = idr_torch.rank
    local_rank = idr_torch.local_rank
    world_size = idr_torch.size
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])
    n_gpu_per_node = world_size // n_nodes
    is_master = global_rank == 0
    
    # Print all task information
    print('{:>2}> Task {:>2} in {:>2} | Node {} in {} | GPU {} in {} | {}'.format(
        global_rank, global_rank, world_size, node_id, n_nodes, local_rank, n_gpu_per_node, 'Master' if is_master else '-'))
    
    # Set the device to use
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')
    
    # Init the process group
    torch.distributed.init_process_group(
        init_method='env://',
        backend='nccl',
        world_size=world_size, 
        rank=global_rank
    )

    return global_rank, local_rank, world_size, is_master, device


def get_infos(hardware, computer):
    if hardware == 'multi-gpu':
        if computer == 'jeanzay':
            global_rank, local_rank, world_size, is_master, device = init()
        else:
            raise Exception('Multi-GPU is only supported on JeanZay')
    else:
        global_rank, local_rank, world_size, is_master = 0, 0, 1, True
        device = torch.device('cuda') if hardware == 'mono-gpu' else torch.device('cpu')
    
    return global_rank, local_rank, world_size, is_master, device


def adapt_to_parallel_computing(hardware, model, local_rank):
    if hardware == 'multi-gpu':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    return model


def get_data_stuff(hardware, train_dataset, eval_dataset, batch_size, nb_workers, world_size, global_rank):
 # CREATE SAMPLERS
    if hardware == 'multi-gpu':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=global_rank
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    
    # CREATE DATALOADERS
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size//world_size,
        drop_last=True,
        num_workers=nb_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size//world_size,
        num_workers=nb_workers,
        pin_memory=True,
        sampler=eval_sampler
    )

    return train_sampler, eval_sampler, train_loader, eval_loader