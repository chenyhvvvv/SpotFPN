import tifffile
import numpy as np
from torch import nn
import torch
import einops
from torch.utils.data import DataLoader
from tqdm import tqdm
# from fpn.factory import make_spot_fpn_resnet
print("======> Alation Study version")
from fpn.ablation_study import make_spot_fpn_resnet
from networks import DeconvNet


class VisiumHDFPNModel(nn.Module):
    def __init__(self, manager, gene_map_shape):
        super(VisiumHDFPNModel, self).__init__()
        self.manager = manager
        self.opt = manager.get_opt()
        self.logger = manager.get_logger()

        # You must use cuda
        self.device = torch.device("cuda")

        self.mask_size = (gene_map_shape[0], gene_map_shape[1])
        self.gene_num = gene_map_shape[2]

        self.basis = tifffile.imread(self.opt.basis)
        # To tensor
        self.basis = torch.from_numpy(self.basis).to(torch.float32).to(self.device)
        self.logger.info(f"Basis shape: {self.basis.shape}")



        self._build_network()
        self._get_optimizer()

    def _build_network(self):
        model_fpn = make_spot_fpn_resnet(out_size=(self.opt.patch_size, self.opt.patch_size),
                                         in_channels=self.gene_num, num_classes=self.opt.deconv_emb_dim)
        self.backbone= model_fpn[0].to(self.device)
        self.fpn = model_fpn[1][0].to(self.device)

        self.deconv = DeconvNet(gene_num=self.gene_num, hidden_dims=self.opt.deconv_emb_dim, n_celltypes=self.basis.shape[0]).to(self.device)
        self.decoder = torch.nn.Sequential(
                            torch.nn.Linear(self.opt.deconv_emb_dim, self.opt.deconv_emb_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.opt.deconv_emb_dim, self.gene_num)
                        ).to(self.device)

    def _get_optimizer(self):
        self.optimizer_bb = torch.optim.Adam(self.backbone.parameters(), lr=self.opt.lr_bb)
        self.optimizer_fpn = torch.optim.Adam(self.fpn.parameters(), lr=self.opt.lr_fpn)
        self.optimizer_deconv = torch.optim.Adam(self.deconv.parameters(), lr=self.opt.lr_deconv)
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr_decoder)
        self.optimizers = [self.optimizer_bb, self.optimizer_fpn, self.optimizer_deconv, self.optimizer_decoder]

    def forward(self, x):
        return 0

    def train_model(self, train_dataset):
        print("Length of train dataset: ", len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        # Training
        self.train()
        for epoch in range(self.opt.epochs):
            for i, batch in enumerate(train_loader):
                # expr_torch, expr_norm_torch, nucl_torch, mask_torch, bg_mask_torch, coords_h1, coords_w1
                expr, expr_norm, nucl, binary_nucl, bg_mask, coords_h1, coords_w1 = batch

                expr = expr.to(self.device)
                expr_norm = expr_norm.to(self.device)
                nucl = nucl.to(self.device)

                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                # Forward
                backbone_out = self.backbone(expr_norm)
                fpn_out = self.fpn(backbone_out)
                feat_map = fpn_out[self.opt.fpn_level]

                # For the smallest feature map
                raw_size = expr.shape[-1]
                feat_map_size = feat_map.shape[-1]
                factor = raw_size // feat_map_size

                # Shape of expr_bin_sum: (batch_size, gene_num, feat_map_size, feat_map_size)
                expr_bin_sum = torch.zeros((expr.shape[0], self.gene_num, feat_map_size, feat_map_size)).to(self.device)
                binary_nucl_sum = torch.zeros((expr.shape[0], feat_map_size, feat_map_size)).to(self.device)
                for i_ in range(feat_map_size):
                    for j_ in range(feat_map_size):
                        expr_bin_sum[:, :, i_, j_] = expr[:, :, i_*factor:(i_+1)*factor, j_*factor:(j_+1)*factor].sum(dim=(-1, -2))
                        binary_nucl_sum[:, i_, j_] = binary_nucl[:, i_*factor:(i_+1)*factor, j_*factor:(j_+1)*factor].sum(dim=(-1, -2))

                # Flatten
                expr_bin_sum_flatten = einops.rearrange(expr_bin_sum, 'b c h w -> (b h w) c')
                feat_map_flatten = einops.rearrange(feat_map, 'b c h w -> (b h w) c')
                binary_nucl_flatten = einops.rearrange(binary_nucl_sum, 'b h w -> (b h w)')
                # Without nucl: Not include in training
                expr_bin_sum_flatten = expr_bin_sum_flatten[binary_nucl_flatten > 0]
                feat_map_flatten = feat_map_flatten[binary_nucl_flatten > 0]
                # if expr_bin_sum_flatten.shape[0] == 0:
                #     continue

                library_size = expr_bin_sum_flatten.sum(dim=-1).unsqueeze(-1)

                loss_deconv = self.deconv(feat_map_flatten, expr_bin_sum_flatten, library_size, self.basis)

                # Reconstruction
                z = self.decoder(feat_map_flatten)
                loss_decoder = torch.nn.functional.mse_loss(z, expr_bin_sum_flatten) * self.opt.recon_loss_weight

                loss = loss_deconv + loss_decoder

                loss.backward()
                # Clip gradient
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.opt.gradient_clip)

                for optimizer in self.optimizers:
                    optimizer.step()

                self.logger.info(f'Epoch {epoch}, iter {i}, deconv_loss {loss_deconv.item()}, decoder_loss {loss_decoder.item()}, valid spot {z.shape[0]}/{binary_nucl_flatten.shape[0]}')

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, saving model...")
                self.save(self.manager.get_checkpoint_dir() + '/model_epoch_{}.pth'.format(epoch))

        # Save the final model
        self.save(self.manager.get_checkpoint_dir() + '/model_final.pth')
        self.logger.info(f"Model saved to {self.manager.get_checkpoint_dir() + '/model_final.pth'}")

    def predict(self, test_dataset):
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=0)

        raw_size_latent_map = np.zeros((self.mask_size[0], self.mask_size[1], self.opt.deconv_emb_dim))
        raw_size_deconv_beta_map = np.zeros((self.mask_size[0], self.mask_size[1], self.basis.shape[0]))

        latent_map = None
        deconv_beta_map = None

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                expr, expr_norm, nucl, binary_nucl, bg_mask, coords_h1, coords_w1 = batch

                expr = expr.to(self.device)
                expr_norm = expr_norm.to(self.device)
                backbone_out = self.backbone(expr_norm)
                fpn_out = self.fpn(backbone_out)
                feat_map = fpn_out[self.opt.fpn_level]

                # For the smallest feature map
                raw_size = expr.shape[-1]
                feat_map_size = feat_map.shape[-1]
                factor = raw_size // feat_map_size

                if latent_map is None:
                    latent_map = np.zeros((self.mask_size[0]//factor + 1, self.mask_size[1]//factor + 1, self.opt.deconv_emb_dim))
                    deconv_beta_map = np.zeros((self.mask_size[0]//factor + 1, self.mask_size[1]//factor + 1, self.basis.shape[0]))

                # Shape of expr_bin_sum: (batch_size, gene_num, feat_map_size, feat_map_size)
                expr_bin_sum = torch.zeros((expr.shape[0], self.gene_num, feat_map_size, feat_map_size)).to(self.device)
                binary_nucl_sum = torch.zeros((expr.shape[0], feat_map_size, feat_map_size)).to(self.device)
                for i in range(feat_map_size):
                    for j in range(feat_map_size):
                        expr_bin_sum[:, :, i, j] = expr[:, :, i*factor:(i+1)*factor, j*factor:(j+1)*factor].sum(dim=(-1, -2))
                        binary_nucl_sum[:, i, j] = binary_nucl[:, i*factor:(i+1)*factor, j*factor:(j+1)*factor].sum(dim=(-1, -2))
                # Flatten
                expr_bin_sum_flatten = einops.rearrange(expr_bin_sum, 'b c h w -> (b h w) c')
                feat_map_flatten = einops.rearrange(feat_map, 'b c h w -> (b h w) c')
                binary_nucl_flatten = einops.rearrange(binary_nucl_sum, 'b h w -> (b h w)')

                library_size = expr_bin_sum_flatten.sum(dim=-1).unsqueeze(-1)
                beta, alpha = self.deconv.deconv(feat_map_flatten)
                # Those without nucl are set to zero
                feat_map_flatten[binary_nucl_flatten == 0] = 0
                beta[binary_nucl_flatten == 0] = 0

                beta = einops.rearrange(beta, '(b h w) c -> b h w c', b=expr.shape[0], h=feat_map_size, w=feat_map_size)
                latent = einops.rearrange(feat_map_flatten, '(b h w) c -> b h w c', b=expr.shape[0], h=feat_map_size, w=feat_map_size)

                # Update latent map and deconv beta map
                for b in range(expr.shape[0]):
                    latent_map[coords_h1[b]//factor:coords_h1[b]//factor+feat_map_size, coords_w1[b]//factor:coords_w1[b]//factor+feat_map_size] = latent[b].cpu().numpy()
                    deconv_beta_map[coords_h1[b]//factor:coords_h1[b]//factor+feat_map_size, coords_w1[b]//factor:coords_w1[b]//factor+feat_map_size] = beta[b].cpu().numpy()

                # Update raw size latent map and deconv beta map
                # Enlarge the latent map and deconv beta map
                for b in range(expr.shape[0]):
                    enlarge_latent = einops.repeat(latent[b].cpu().numpy(), 'h w c -> (h f1) (w f2) c', f1=factor, f2=factor)
                    enlarge_beta = einops.repeat(beta[b].cpu().numpy(), 'h w c -> (h f1) (w f2) c', f1=factor, f2=factor)
                    raw_size_latent_map[coords_h1[b]:coords_h1[b] + raw_size, coords_w1[b]:coords_w1[b] + raw_size] = enlarge_latent
                    raw_size_deconv_beta_map[coords_h1[b]:coords_h1[b] + raw_size, coords_w1[b]:coords_w1[b] + raw_size] = enlarge_beta

        # Save
        tifffile.imsave(self.manager.get_log_dir() + '/latent_map.tif', latent_map)
        tifffile.imsave(self.manager.get_log_dir() + '/deconv_beta_map.tif', deconv_beta_map)
        tifffile.imsave(self.manager.get_log_dir() + '/raw_size_latent_map.tif', raw_size_latent_map)
        tifffile.imsave(self.manager.get_log_dir() + '/raw_size_deconv_beta_map.tif', raw_size_deconv_beta_map)

    def save(self, path):
        """Save the model state
        """
        save_dict = {'model_state': self.state_dict()}

        # for k, v in self.schedulers.items():
        #     save_dict[k + '_state'] = v.state_dict()
        torch.save(save_dict, path)

    def load(self, path, strict = True):
        """Load a model state from a checkpoint file
        """
        checkpoint_file = path
        checkpoints = torch.load(checkpoint_file)
        self.load_state_dict(checkpoints['model_state'], strict=strict)
