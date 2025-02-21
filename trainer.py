from dataset import *
from model import Image_Augmenter
from parallel_computing_utils import *
from img_utils import get_reverse_transform

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from time import time


if __name__ == "__main__":

    writer = SummaryWriter()

    batch_size = 512
    epochs_nb = 1000
    model_path = "models/best_model.pt"
    lr = 1e-4
    # load_model = True
    # warmup_epochs_nb = 0
    load_model = False
    warmup_epochs_nb = 10

    hardware = "mono-gpu"
    # hardware = "cpu"
    computer = "pagoda"
    nb_workers = 20

    global_rank, local_rank, world_size, is_master, device = get_infos(hardware, computer)
    if hardware != 'cpu': torch.backends.cudnn.benchmark = True

    model = Image_Augmenter(cifar=True)
    model = adapt_to_parallel_computing(hardware, model, local_rank)
    model.to(device)
    if load_model:
        model.load_weights(model_path)
        print("Model loaded")
    
    train_dataset = get_dataset(device, "cifar", 'train')
    eval_dataset = get_dataset(device, "cifar", 'eval')
    reverse_transform = get_reverse_transform(train_dataset.no_transform)

    train_sampler, eval_sampler, train_loader, eval_loader = get_data_stuff(hardware, train_dataset, eval_dataset,
                                                                            batch_size, nb_workers, world_size, global_rank)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(),
                    lr=lr)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1,
        total_iters = warmup_epochs_nb * len(train_loader),
        last_epoch = -1
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = (epochs_nb - warmup_epochs_nb) * len(train_loader),
        last_epoch = -1,
        eta_min = 1e-8
    )
    
    best_loss_value = 1000

    train_loader.dataset.update_tau(0.)
    eval_loader.dataset.update_tau(0.)

    for epoch in range(1, epochs_nb+1):
        ### TRAIN ###
        losses = []
        model.train()
        start = time()
        for i, batch in enumerate(train_loader):
            img_0, img_t, params = batch

            img_0 = img_0.to(device)
            img_t = img_t.to(device)
            params = params.to(device)
        
            img_out = model(img_0, params)
            
            loss = criterion(img_out, img_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.float())

             # inside the batch loop because the "scheduler size" = epoch_nb * len(train_dataloader)
            if epoch <= warmup_epochs_nb:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            if i%10==0:
                print(f"Epoch {epoch} - batch {i+1}/{len(train_loader)}")
            
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('train_MSE', avg_loss, epoch)
        print(f"Epoch {epoch} - train loss: {avg_loss:2.5f} - duration: {time() - start:5.5f}s")

        img_0 = reverse_transform(img_0[0])
        img_t = reverse_transform(img_t[0])
        img_out = reverse_transform(img_out[0])
        writer.add_image(f"train source epoch {epoch}", img_0)
        writer.add_image(f"train target epoch {epoch}", img_t)
        writer.add_image(f"train output epoch {epoch}", img_out)
        
        img_save = torch.stack((img_0, img_t, img_out), dim=0)
        save_image(img_save,
                   f'img/train images epoch {epoch}.jpg')

        ### TEST ###
        with torch.no_grad():
            model.eval()
            losses = []
            start = time()
            for i, batch in enumerate(eval_loader):
                img_0, img_t, params = batch

                img_0 = img_0.to(device)
                img_t = img_t.to(device)
                params = params.to(device)

                img_out = model(img_0, params)
                
                loss = criterion(img_out, img_t)
                losses.append(loss.float())

                if i%20==0:
                    print(f"Epoch {epoch} - batch {i+1}/{len(eval_loader)}")

        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('eval_MSE', avg_loss, epoch)
        print(f"Epoch {epoch} - test loss : {avg_loss:2.5f} - duration: {time() - start:5.5f}s")

        img_0 = reverse_transform(img_0[0])
        img_t = reverse_transform(img_t[0])
        img_out = reverse_transform(img_out[0])
        writer.add_image(f"train source epoch {epoch}", img_0)
        writer.add_image(f"train target epoch {epoch}", img_t)
        writer.add_image(f"train output epoch {epoch}", img_out)
        
        img_save = torch.stack((img_0, img_t, img_out), dim=0)
        save_image(img_save,
                   f'img/test images epoch {epoch}.jpg')

        if avg_loss < best_loss_value:
            model.save(model_path)
            best_loss_value = avg_loss
            print("### Model saved ###")

