import torch

from torch import nn
from torch.nn import functional as F
from utils.torch_utils import export_to_netron
import math
import logging

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        # TODO: Read ResNet paper again https://arxiv.org/abs/1512.03385
        layers = []
        for _ in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        # ResNet V1-style.
        output = self.relu(h)
        return output



class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        kernel_size = 4,
        stride = 2,
        padding = 1,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        # The last ReLU from the Sonnet example is omitted because ResidualStack starts
        # off with a ReLU.
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())

        conv.add_module(
            "final_conv",
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_hiddens,
        num_upsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        kernel_size = 4,
        stride = 2,
        padding = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

            else:
                (in_channels, out_channels) = (num_hiddens // 2, 3)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    # This module keeps track of a hidden exponential moving average that is
    # initialized as a vector of zeros which is then normalized to give the average.
    # This gives us a moving average which isn't biased towards either zero or the
    # initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders

    Taken from https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size
            logging.info(f"Initialized feature pool with {self.features.shape} features")

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features

    

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon, online_update=False, anchor="random"):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon


        # Online update to prevent codebook collapse
        # Adapted from CVQ-VAE: https://arxiv.org/abs/2307.15139
        self.online_update = online_update
        self.anchor = anchor
        self.pool = FeaturePool(self.num_embeddings, self.embedding_dim)
        self.register_buffer("embed_prob", torch.zeros(self.num_embeddings))

        # Flag to ensure online update happens only after the first training step
        self.register_buffer("first_update_done", torch.tensor(False))

        # Dictionary embeddings.
        scale = 1.0
        limit = math.sqrt(3.0 * scale / self.embedding_dim) # equivalent to having variance 1/embedding_dim (see: https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )

        if use_ema:
            logging.info(f"Using EMA for codebook updates with decay {decay}")
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            logging.info("Using non-EMA for codebook updates")
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

        # Attributes to store auxiliary outputs for compatibility with torchscan
        self.last_dictionary_loss = None
        self.last_commitment_loss = None
        self.last_encoding_indices = None
        self.last_flat_x = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.embedding_dim

        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        assert flat_x.shape == (B * H * W, self.embedding_dim)

        self.last_flat_x = flat_x
        
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        
        assert encoding_indices.shape == (B * H * W,)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        assert quantized_x.shape == (B, C, H, W)

        # Calculate losses using the non-straight-through quantized_x
        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            # EMA loss is implicitly handled by the embedding updates
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()

        # Apply Straight-through gradient. See Section 3.2.
        # This tensor is passed downstream.
        quantized_x = x + (quantized_x - x).detach()

        encoding_one_hots = F.one_hot(encoding_indices, self.num_embeddings).type(flat_x.dtype)
        assert encoding_one_hots.shape == (B * H * W, self.num_embeddings)

        avg_probs = torch.mean(encoding_one_hots, dim=0)
        
        if self.training:
            if self.use_ema:
                with torch.no_grad():
                    # See Appendix A.1 of "Neural Discrete Representation Learning".
                    # Cluster counts.
                    n_i_ts = encoding_one_hots.sum(0)
                    # Updated exponential moving average of the cluster counts.
                    # See Equation (6).
                    self.N_i_ts(n_i_ts)

                    # Exponential moving average of the embeddings. See Equation (7).
                    embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                    self.m_i_ts(embed_sums)

                    # Update dictionary embeddings. See Equation (8).
                    # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                    N_i_ts_sum = self.N_i_ts.average.sum()
                    N_i_ts_stable = (
                        (self.N_i_ts.average + self.epsilon)
                        / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                        * N_i_ts_sum
                    )
                    # Update the buffer directly
                    self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)
                            
            # Online update logic, executed only after the first training step
            if self.first_update_done:
                if self.online_update:
                    self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
                    # decay parameter based on the average usage
                    decay = torch.exp(-(self.embed_prob*self.num_embeddings*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embedding_dim)
                    if self.anchor == "random":
                        random_feat = self.pool.query(flat_x.detach())
                        self.e_i_ts.data = self.e_i_ts.data * (1 - decay.t()) + random_feat.t() * decay.t()
                    
                    # TODO: Implement other anchor strategies
            else:
                # Set the flag to True after the first training forward pass
                # Ensures online update starts from the second training step onwards
                self.first_update_done = torch.tensor(True, device=x.device)


        # Store auxiliary outputs as attributes
        self.last_dictionary_loss = dictionary_loss
        self.last_commitment_loss = commitment_loss
        self.last_encoding_indices = encoding_indices.view(x.shape[0], -1)

        return quantized_x


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        embedding_dim,
        num_embeddings,
        use_ema,
        decay,
        epsilon,
        online_update,
        anchor,
        kernel_size = 4,
        stride = 2,
        padding = 1,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
            kernel_size,
            stride,
            padding,
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon, online_update, anchor
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
            kernel_size,
            stride,
            padding,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        # self.vq.forward now returns only the quantized tensor
        z_quantized = self.vq(z)
        # Retrieve the losses and indices from the vq module's attributes
        self.encoding_indices = self.vq.last_encoding_indices
        self.dictionary_loss = self.vq.last_dictionary_loss
        self.commitment_loss = self.vq.last_commitment_loss
        self.encoding_indices = self.vq.last_encoding_indices
        return z_quantized

    def forward(self, x):
        z_quantized = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return x_recon


if __name__ == "__main__":
    # Create a sample model
    model_args = {
        "in_channels": 3,
        "num_hiddens": 256,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 256,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": True,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to("cuda")

    use_onnx = False

    if use_onnx:
        from utils.torch_utils import export_to_netron
  
        # Visualize the model graph
        dummy_input = torch.randn(1, 3, 32, 32).to("cuda")  # Assuming 3 channel 64x64 images
        export_to_netron(model, dummy_input, "../../vqvae_model.onnx")
    

    use_torchscan = True

    if use_torchscan:
        from torchscan import summary
        summary(model, (3, 32, 32))
