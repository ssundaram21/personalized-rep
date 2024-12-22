import logging
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.personal_model import PersonalModel
from dataset.triplet_dataset import TripletDataset
from util.data_util import makedirs
from util.transforms import get_transform
from util.misc import Mean, seed_worker
from util import pidfile
from contrastive.losses import get_loss_fn
from peft import get_peft_model, LoraConfig
import os

# Set up logging
log = logging.getLogger("lightning.pytorch")
log.propagate = False
log.setLevel(logging.INFO)

class LossLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.train_scores = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends. Saves the losses and scores.

        Args:
            trainer: The trainer instance.
            pl_module: The Lightning module being trained.
        """
        train_score = pl_module.train_metrics['score'].compute()
        train_loss = pl_module.train_metrics['loss'].compute()

        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_score is not None:
            self.train_scores.append(train_score.item())

        log_dir = trainer.log_dir
        train_loss_path = os.path.join(log_dir, 'train_losses.txt')
        train_score_path = os.path.join(log_dir, 'train_scores.txt')

        with open(train_loss_path, 'a') as f:
            f.write(f"{train_loss.item()}\n")
        with open(train_score_path, 'a') as f:
            f.write(f"{train_score.item()}\n")

class LightningPersonalModel(pl.LightningModule):
    def __init__(self, 
                 train_model_type: str = "dinov2_vitb14", 
                 hidden_size: int = 512, 
                 train_loss_fn: str = "hinge",
                 use_lora: bool = False, 
                 patch_tokens: bool = False,
                 cache_dir: str = "./cache", 
                 train_lr: float = 0.03, 
                 train_margin: float = 0.05, 
                 train_lora_r: int = 16, 
                 train_lora_alpha: float = 0.5, 
                 train_lora_dropout: float = 0.3, 
                 train_weight_decay: float = 0.0, 
                 train_data_len: int = 1,
                 device: str = "cuda",
                 **kwargs):
        """
        Initializes the LightningPersonalModel.

        Args:
            train_model_type (str): Type of the backbone model.
            hidden_size (int): Size of the hidden layer.
            train_loss_fn (str): Loss function to use.
            use_lora (bool): Whether to use LoRA.
            patch_tokens (bool): Whether to patch tokens.
            cache_dir (str): Directory to load the pretrained model from.
            train_lr (float): Learning rate.
            train_margin (float): Margin for the loss function.
            train_lora_r (int): LoRA rank.
            train_lora_alpha (float): LoRA alpha.
            train_lora_dropout (float): LoRA dropout.
            train_weight_decay (float): Weight decay.
            train_data_len (int): Length of the training data.
            device (str): Device to use for training.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_type = train_model_type
        self.hidden_size = hidden_size
        self.use_lora = use_lora
        self.lr = train_lr
        self.margin = train_margin
        self.weight_decay = train_weight_decay
        self.lora_r = train_lora_r
        self.lora_alpha = train_lora_alpha
        self.lora_dropout = train_lora_dropout
        self.train_data_len = train_data_len

        self.personal_model = PersonalModel(
            model_type=self.model_type, 
            hidden_size=self.hidden_size, 
            lora=self.use_lora, 
            patch_tokens=patch_tokens,
            load_dir=cache_dir, 
            device=device
        )

        self.train_metrics = {'loss': Mean().to(device), 'score': Mean().to(device)}
        self.__reset_train_metrics()

        if self.use_lora:
            self.__prep_lora_model()
        else:
            self.__prep_linear_model()

        pytorch_total_params = sum(p.numel() for p in self.personal_model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in self.personal_model.parameters() if p.requires_grad)
        print(f'Total params: {pytorch_total_params} | Trainable params: {pytorch_total_trainable_params} '
              f'| % Trainable: {pytorch_total_trainable_params / pytorch_total_params * 100}')

        self.train_loss_fn = train_loss_fn
        self.criterion = get_loss_fn(train_loss_fn, device=device, margin=self.margin)
        self.epoch_loss_train = 0.0
        self.train_num_correct = 0.0

    def get_dist(self, embed_0, embed_1):
        """
        Computes the cosine similarity distance between two embeddings.

        Args:
            embed_0: First embedding.
            embed_1: Second embedding.

        Returns:
            Cosine similarity distance.
        """
        return 1 - F.cosine_similarity(embed_0, embed_1, dim=-1)

    def forward(self, img_ref, img_0, img_1):
        """
        Forward pass through the model.

        Args:
            img_ref: Reference image.
            img_0: First image.
            img_1: Second image.

        Returns:
            Embeddings for the reference, first, and second images.
        """
        emb_ref = self.personal_model(img_ref)
        emb_0 = self.personal_model(img_0)
        emb_1 = self.personal_model(img_1)
        return emb_ref, emb_0, emb_1

    def forward_loss(self, img_ref, img_0, img_1, target):
        """
        Computes the loss for the forward pass.

        Args:
            img_ref: Reference image.
            img_0: First image.
            img_1: Second image.
            target: Target labels indicating where the positive images are.

        Returns:
            Loss and number of correct predictions.
        """
        emb_ref, emb_0, emb_1 = self.forward(img_ref, img_0, img_1)
        if self.train_loss_fn == "hinge":
            dist_0 = self.get_dist(img_ref, img_0)
            dist_1 = self.get_dist(img_ref, img_1)
            decision = torch.lt(dist_1, dist_0)
            logit = dist_0 - dist_1
            loss = self.criterion(logit.squeeze(), target)
            train_num_correct = ((target == 1) == decision).sum()
        elif "info_nce" in self.train_loss_fn:
            positive_emb = torch.where((target == 0).unsqueeze(1), emb_0, emb_1)
            negative_emb = torch.where((target == 1).unsqueeze(1), emb_0, emb_1)
            loss = self.criterion(emb_ref, positive_emb, negative_emb)
            sim_pos = F.cosine_similarity(emb_ref, positive_emb)
            sim_neg = F.cosine_similarity(emb_ref, negative_emb)
            train_num_correct = (sim_pos > sim_neg).sum()
        return loss, train_num_correct

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            Loss for the batch.
        """
        img_ref, img_0, img_1, target, idx = batch
        loss, train_num_correct = self.forward_loss(img_ref, img_0, img_1, target)
        self.train_metrics['loss'].update(loss, target.shape[0])
        self.train_metrics['score'].update(train_num_correct, target.shape[0])
        self.train_num_correct += train_num_correct        
        self.epoch_loss_train += loss.item() / target.shape[0]
        return loss

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        """
        self.epoch_loss_train = 0.0
        self.train_num_correct = 0.0
        self.visualize = True
        self.started = True

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        epoch = self.current_epoch + 1 if self.started else 0
        self.logger.experiment.add_scalar(f'train_loss/', self.epoch_loss_train / self.trainer.num_training_batches, epoch)
        self.logger.experiment.add_scalar(f'train_acc/', self.train_num_correct / self.train_data_len, epoch)

        print(f"Train loss {self.epoch_loss_train / self.trainer.num_training_batches}, train score {self.train_num_correct / self.train_data_len}")

    def on_train_start(self):
        """
        Called at the start of training.
        """
        self.personal_model.train()

    def configure_optimizers(self):
        """
        Configures the optimizers for training.

        Returns:
            List of optimizers.
        """
        params = list(self.personal_model.model.parameters())
        params += list(self.personal_model.mlp.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        return [optimizer]

    def __reset_train_metrics(self):
        """
        Resets the training metrics.
        """
        for k, v in self.train_metrics.items():
            v.reset()

    def __prep_lora_model(self):
        """
        Prepares the model for LoRA training.
        """
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias='none',
            target_modules=['qkv']
        )
        self.personal_model.config = config
        self.personal_model = get_peft_model(self.personal_model, config)

    def __prep_linear_model(self):
        """
        Prepares the model for linear training.
        """
        self.personal_model.model.requires_grad_(False)
        self.personal_model.mlp.requires_grad_(True)

def train(args, device):
    """
    Trains the model using the provided arguments and device.

    Args:
        args: Arguments for training.
        device: Device to use for training.

    Returns:
        checkpoint_root: Path to the checkpoint root directory.
    """
    print()
    ckpt_dir = args.ckpt_dir
    makedirs(ckpt_dir)
    print(f"Seeding with {args.seed}")
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Setup triplet datasets
    preprocess = get_transform(args.train_model_type)

    train_dataset = TripletDataset(
        csv_path=args.dataset_csv, 
        preprocess=preprocess, 
        augment=args.train_augment
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        num_workers=args.train_num_workers, 
        shuffle=True,                      
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    logger = TensorBoardLogger(save_dir=ckpt_dir, default_hp_metric=False)

    # Setup PyTorch Lightning trainer
    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        log_every_n_steps=10,
        logger=logger,
        max_epochs=args.train_epochs,
        default_root_dir=ckpt_dir,
        callbacks=[ModelCheckpoint(every_n_epochs=2,
                                   save_top_k=-1,
                                   save_last=True,
                                   filename='{epoch:02d}'),
                   LossLogger()],
        num_sanity_val_steps=0
    )

    # Setup checkpoint root and save config
    checkpoint_root = os.path.join(ckpt_dir, "lightning_logs", f'version_{trainer.logger.version}')
    os.makedirs(checkpoint_root, exist_ok=True)
    pidfile.pidfile_taken(os.path.join(checkpoint_root, 'lockfile.pid'))
    with open(os.path.join(checkpoint_root, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    logging.basicConfig(filename=os.path.join(checkpoint_root, 'exp.log'), level=logging.INFO, force=True)
    logging.info("Arguments: %s", vars(args))

    # Initialize Lightning model and train
    model = LightningPersonalModel(device=device, train_data_len=len(train_dataset), **vars(args))
    logging.info("Training")
    trainer.fit(model, train_loader)

    pidfile.mark_job_done(checkpoint_root)
    print("Done :)")
    return checkpoint_root
