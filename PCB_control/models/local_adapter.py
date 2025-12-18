import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,  # 根据输入的维度参数 dims，动态创建并返回对应的 1D、2D 或 3D 卷积层
    linear,
    zero_module,  # 把一个模块的所有参数初始化为 0，并返回这个模块本身
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepBlock, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.util import exists


class LocalTimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, local_features=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, LocalResBlock):
                x = layer(x, emb, local_features)
            else:
                x = layer(x)
        return x


class FDN(nn.Module):  # 这里可以改
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        # self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        # self.conv_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(
                label_nc, label_nc,
                kernel_size=ks,
                padding=pw,
                groups=label_nc
            ),  # Depth-wise
            nn.Conv2d(label_nc, norm_nc,kernel_size=1)  # Point-wise
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(
                label_nc, label_nc,
                kernel_size=ks,
                padding=pw,
                groups=label_nc
            ),  # Depth-wise
            nn.Conv2d(label_nc, norm_nc,kernel_size=1)  # Point-wise
        )

    def forward(self, x, local_features):
        normalized = self.param_free_norm(x)
        assert local_features.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(local_features)
        beta = self.conv_beta(local_features)
        out = normalized * (1 + gamma) + beta
        return out


class LocalResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        inject_channels=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.norm_in = FDN(channels, inject_channels)
        self.norm_out = FDN(self.out_channels, inject_channels)

        self.in_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, local_conditions):
        return checkpoint(
            self._forward, (x, emb, local_conditions), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, local_conditions):
        h = self.norm_in(x, local_conditions)
        h = self.in_layers(h)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.norm_out(h, local_conditions)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h

# *****************************************************************
class SpaceToDepth(nn.Module):  # 空间映射到通道
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0

        x = x.view(
            B, C,
            H // bs, bs,
            W // bs, bs
        )
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * bs * bs, H // bs, W // bs)
        return x

class SPDConv(nn.Module):  # SPD卷积
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        block_size=2,
        padding=1,
        bias=True
    ):
        super().__init__()
        self.spd = SpaceToDepth(block_size)
        self.conv = nn.Conv2d(
            in_channels * block_size * block_size,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        x = self.spd(x)
        x = self.conv(x)
        return x

class DSConv(nn.Module):  # 深度可分离卷积
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    ):
        super().__init__()
        # depth-wise conv
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        # point-wise conv
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
# *****************************************************************

class FeatureExtractor(nn.Module):
    def __init__(self, local_channels, inject_channels, dims=2):
        super().__init__()
        # self.pre_extractor = LocalTimestepEmbedSequential(
        #     conv_nd(dims, local_channels, 32, 3, padding=1),  # 0
        #     nn.SiLU(),  # 1
        #     conv_nd(dims, 32, 64, 3, padding=1, stride=2),  # 2
        #     nn.SiLU(),  # 3
        #     conv_nd(dims, 64, 64, 3, padding=1),  # 4
        #     nn.SiLU(),  # 5
        #     conv_nd(dims, 64, 128, 3, padding=1, stride=2),  # 6
        #     nn.SiLU(),  # 7
        #     conv_nd(dims, 128, 128, 3, padding=1),  # 8
        #     nn.SiLU(),  # 9
        # )
        # self.extractors = nn.ModuleList([
        #     LocalTimestepEmbedSequential(  # 0
        #         conv_nd(dims, 128, inject_channels[0], 3, padding=1, stride=2),
        #         nn.SiLU()
        #     ),
        #     LocalTimestepEmbedSequential(  # 1
        #         conv_nd(dims, inject_channels[0], inject_channels[1], 3, padding=1, stride=2),
        #         nn.SiLU()
        #     ),
        #     LocalTimestepEmbedSequential(  # 2
        #         conv_nd(dims, inject_channels[1], inject_channels[2], 3, padding=1, stride=2),
        #         nn.SiLU()
        #     ),
        #     LocalTimestepEmbedSequential(  # 3
        #         conv_nd(dims, inject_channels[2], inject_channels[3], 3, padding=1, stride=2),
        #         nn.SiLU()
        #     )
        # ])
        self.pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, local_channels, 32, 3, padding=1),  # 0
            nn.SiLU(),  # 1
            # ↓↓↓ 原 stride=2 卷积 → SPD 卷积 ↓↓↓
            SPDConv(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                block_size=2,
                padding=1
            ),  # 2
            nn.SiLU(),  # 3
            conv_nd(dims, 64, 64, 3, padding=1),  # 4
            nn.SiLU(),  # 5
            # ↓↓↓ 原 stride=2 卷积 → SPD 卷积 ↓↓↓
            SPDConv(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                block_size=2,
                padding=1
            ),  # 6
            nn.SiLU(),  # 7
            conv_nd(dims, 128, 128, 3, padding=1),  # 8
            nn.SiLU(),  # 9
        )
        self.extractors = nn.ModuleList([
            LocalTimestepEmbedSequential(  # 0
                DSConv(
                    in_channels=128,
                    out_channels=inject_channels[0],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.SiLU()
            ),
            LocalTimestepEmbedSequential(  # 1
                DSConv(
                    in_channels=inject_channels[0],
                    out_channels=inject_channels[1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.SiLU()
            ),
            LocalTimestepEmbedSequential(  # 2
                DSConv(
                    in_channels=inject_channels[1],
                    out_channels=inject_channels[2],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.SiLU()
            ),
            LocalTimestepEmbedSequential(  # 3
                DSConv(
                    in_channels=inject_channels[2],
                    out_channels=inject_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.SiLU()
            )
        ])
        self.zero_convs = nn.ModuleList([
            zero_module(conv_nd(dims, inject_channels[0], inject_channels[0], 3, padding=1)),  # 0
            zero_module(conv_nd(dims, inject_channels[1], inject_channels[1], 3, padding=1)),  # 1
            zero_module(conv_nd(dims, inject_channels[2], inject_channels[2], 3, padding=1)),  # 2
            zero_module(conv_nd(dims, inject_channels[3], inject_channels[3], 3, padding=1))   # 3
        ])
    
    def forward(self, local_conditions):
        local_features = self.pre_extractor(local_conditions, None)
        assert len(self.extractors) == len(self.zero_convs)
        
        output_features = []
        for idx in range(len(self.extractors)):
            local_features = self.extractors[idx](local_features, None)
            output_features.append(self.zero_convs[idx](local_features))
        return output_features


class LocalAdapter(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            local_channels,
            inject_channels,
            inject_layers,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.inject_layers = inject_layers
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(  # 这里不能改，有预训练参数
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.feature_extractor = FeatureExtractor(local_channels, inject_channels)  # 这里可以改，全部是初始化参数
        self.input_blocks = nn.ModuleList(
            [
                LocalTimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                if (1 + 3*level + nr) in self.inject_layers:
                    layers = [
                        LocalResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            inject_channels=inject_channels[level]
                        )
                    ]
                else:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(LocalTimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    LocalTimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = LocalTimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return LocalTimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, timesteps, context, local_conditions, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        local_features = self.feature_extractor(local_conditions)

        outs = []
        h = x.type(self.dtype)
        for layer_idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if layer_idx in self.inject_layers:
                h = module(h, emb, context, local_features[self.inject_layers.index(layer_idx)])
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class LocalControlUNetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, local_control=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        h += local_control.pop()

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop() + local_control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)