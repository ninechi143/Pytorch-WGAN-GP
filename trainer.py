import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from WGAN_GP_dataset import train_dataset , normalize
from WGAN_GP_model import Generator , Discriminator
from WGAN_GP_loss import WGAN_GP_Disc_Loss , WGAN_GP_Gen_Loss , Hinge_Disc_Loss

from time import perf_counter

class WGAN_GP_trainer():

    def __init__(self,args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.Lambda = args.Lambda
        self.normalize = args.normalize
        self.resume = args.resume
        self.use_log = args.log

        self.start_epoch = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}")


    def load_data(self):

        print("[!] Data Loading...")


        self.train_dataset = train_dataset()
        self.data_statistics = self.train_dataset.get_statistics()


        transforms_list = [torchvision.transforms.Resize(64)]
        if self.normalize: 
            transforms_list.append(
                        normalize(self.data_statistics[0] , self.data_statistics[1]))

        self.train_dataset.set_transforms(
                        torchvision.transforms.Compose(transforms_list))
        

        self.train_loader = DataLoader(dataset = self.train_dataset,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       num_workers = 1)

        print("[!] Data Loading Done.")


    def setup(self):

        print("[!] Setup...")

        self.data_for_log = torch.randn(64 , 100 , 1 , 1).to(self.device)
        self.log_writer = SummaryWriter('logs') if self.use_log else None
        
        # define our model, loss function, and optimizer
        self.Generator = Generator().to(self.device)
        self.Discriminator = Discriminator().to(self.device)

  
        if self.optim.lower() == "adam":
            self.optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=self.lr , betas = (0. , 0.9))
            self.optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr= self.lr, betas = (0., 0.9))
        else:
            self.optimizer_G = torch.optim.RMSprop(self.Generator.parameters(), lr = self.lr)
            self.optimizer_D = torch.optim.RMSprop(self.Discriminator.parameters(), lr = self.lr)


        self.criterion_G = WGAN_GP_Gen_Loss()
        self.criterion_G.to(self.device)

        self.criterion_D = WGAN_GP_Disc_Loss(self.Lambda , self.device)
        # self.criterion_D = Hinge_Disc_Loss(self.Lambda , self.device)
        self.criterion_D.to(self.device)



        # load checkpoint file to resume training
        if self.resume:
            print(f"[!] Resume training from the file : {self.resume}")
            checkpoint = torch.load(self.resume)
            self.Generator.load_state_dict(checkpoint['model_state'][0])
            self.Discriminator.load_state_dict(checkpoint["model_state"][1])
            try:
                self.start_epoch = checkpoint['epoch']
            except:
                pass

        print("[!] Setup Done.")


    def train(self):

        print("[!] Model training...")
        avg_time = 0
        n_total_steps = len(self.train_loader)


        for epoch in range(self.epochs):

            st = perf_counter()

            for i , data in enumerate(self.train_loader):

                real_images = data.to(self.device)

                # D-training
                for _ in range(3):

                    # feedforward: sample noise and generate images
                    z = torch.randn(real_images.shape[0] , 100 , 1 , 1).to(self.device)            
                    fake_images = self.Generator(z)

                    real_logits = self.Discriminator(real_images)
                    fake_logits = self.Discriminator(fake_images)

                    # compute D-loss
                    alpha = torch.rand(real_images.shape[0] , 1 , 1 ,1).to(self.device)
                    interpolate_images = fake_images + alpha * (real_images - fake_images)
                    interpolate_logits = self.Discriminator(interpolate_images)
                    loss_D = self.criterion_D(real_logits,
                                        fake_logits,
                                        interpolate_logits,
                                        interpolate_images)


                    # D-training updates
                    self.optimizer_D.zero_grad()
                    loss_D.backward()
                    self.optimizer_D.step()


                # G-training and feedforward
                z2 = torch.randn(real_images.shape[0] , 100 , 1 , 1).to(self.device)
                fake_images2 = self.Generator(z2)
                fake_logits2 = self.Discriminator(fake_images2)
                
                # compute G-loss
                loss_G = self.criterion_G(fake_logits2)

                # G-training updates
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()


                # tensorboard: track training process
                if (i+1) % 50 == 0:
                    with torch.no_grad():
                        self.Generator.eval()

                        image_log = self.Generator(self.data_for_log)
                        # image_log = (image_log * self.data_statistics[1] + self.data_statistics[0]) * 255. if self.normalize else image_log * 255.
                        image_grid = torchvision.utils.make_grid(image_log , nrow = 8 , normalize = True)

                        print(f"[!] Epoch : [{epoch+1}], step : [{i+1} / {n_total_steps}]")
                        self.log_writer.add_image("Generator Image" , image_grid , epoch * n_total_steps + i + 1)
                        self.Generator.train()


            avg_time = avg_time + (perf_counter() - st - avg_time) / (epoch+1)
            print(f"[!] Epoch : [{epoch+1}/{self.epochs}] done. Average Training Time: {avg_time:.3f}\n") 


        if self.use_log:
            self.log_writer.close()

        print("[!] Training Done.\n")

    
    def save(self):

        print("[!] Model saving...")
        checkpoint = {"model_state": [self.Generator.state_dict() , self.Discriminator.state_dict()]}
        torch.save(checkpoint , "checkpoint.pth")
        print("[!] Saving Done.")