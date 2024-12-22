import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchstat import stat

import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.manifold import TSNE

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from peft import LoraConfig, get_peft_model
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from peft.tuners.lora.layer import Linear
from clip.GQAtten import MutiGroupAttention
import torch.quantization


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, add_gate=True, gate = None):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.add_gate = add_gate
        if self.add_gate:
            self.gate = gate

    def forward(self, prompts, tokenized_prompts):
        if self.add_gate:
            gate = torch.sigmoid(self.gate(prompts))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
        prompts = gate * prompts
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection



        return x

# Text Prompt
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

# Visual Prompt
class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels  # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers
        if cfg.TRAINER.VPT.PROMPT_DEPTH_VISION == "shallow":
            print("Create shallow prompt (创建浅层VPT)")
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        else:
            print("Create deep prompt (创建深层VPT)")
            ctx_vectors = torch.empty(self.layers, self.n_ctx, self.ctx_dim, dtype=self.dtype)
            for i in range(self.layers):
                nn.init.normal_(ctx_vectors[i], std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self):
        return self.ctx

##### Adapter1
class PositionWiseFeedForward1(nn.Module):
    def __init__(self, embed_dim, d_ff, reduction=8):
        super(PositionWiseFeedForward1, self).__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff // reduction)
        self.fc2 = nn.Linear(d_ff // reduction, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

##### Ligth-Adapter
class PositionWiseFeedForward2(nn.Module):
    def __init__(self, embed_dim, d_ff, reduction=8):
        super(PositionWiseFeedForward2, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.ff(x)

class M_AdapterLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout, add_gate=True, gate = None):
        super(M_AdapterLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attn = MutiGroupAttention(embed_dim, num_heads, group_num=4)
        self.feed_forward = PositionWiseFeedForward2(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.add_gate = add_gate

        if self.add_gate:
            self.gate = gate

    def forward(self, x, y=None):
        self_attn_output = self.self_attn(x, x, x)
        if y is not None:
            cross_attn_output = self.self_attn(x, y, x)
        else:
            cross_attn_output = self.self_attn(x, x, x)

        if  self.add_gate:
            gate1 = torch.sigmoid(self.gate(self_attn_output))
            gate1 = torch.mean(gate1, dim=1).unsqueeze(-1)
            x = self.norm1(x + self.dropout(self_attn_output * gate1) + self.dropout(cross_attn_output* (1-gate1)))
        else:
            x = self.norm1(x + self.dropout(self_attn_output) + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        if self.add_gate:
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)

        ff_output = gate * ff_output

        x = self.norm2(x + self.dropout(ff_output))
        return x

def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class CustomAdapter(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, d_ff=2048, dropout=0.1, add_gate = True, visual_gate= None, text_gate = None):
        super(CustomAdapter, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Gate Attention adapter
        self.image_adapter = M_AdapterLayer(self.embed_dim, self.num_heads, self.d_ff, self.dropout,add_gate=add_gate, gate=visual_gate)
        self.text_adapter = M_AdapterLayer(self.embed_dim, self.num_heads, self.d_ff, self.dropout,add_gate=add_gate, gate=text_gate)

    def forward(self, image_embeds, text_embeds):
        return self.image_adapter(image_embeds, y=text_embeds), self.text_adapter(text_embeds, y=image_embeds)
##### Adapter End

class ProjLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.proj = clip_model.visual.proj

    def forward(self, x):
        if self.proj is not None:
            x = x @ self.proj
        return x

class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model, add_gate=True):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        print("VPT_n_ctx:", self.n_ctx)
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels  # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers
        self.cfg = cfg

        # model
        self.transformer = clip_model.visual.transformer
        self.resblocks: nn.Sequential = self.transformer.resblocks
        self.layers = self.transformer.layers

        self.visual_prompt = VPTDeepPromptLearner(cfg, classnames, clip_model)

        # self.add_gate = add_gate
        # if self.add_gate:
        #     self.gate = nn.Linear(768, 1).half()
        #     self.gate.apply(init_bert_weights)

    def forward(self, x):
        ctx = self.visual_prompt()

        if self.cfg.TRAINER.VPT.PROMPT_DEPTH_VISION == "shallow":
            ctx = ctx.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x[:-self.n_ctx, :, :]
            x = torch.cat([x, ctx], dim=0)
            x = self.transformer(x)

        else:
            ctx = ctx.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
            for i in range(self.layers):
                if i != 0:
                    x = x[:-self.n_ctx, :, :]

                # print(ctx[i].shape, x.shape)
                x = torch.cat([x, ctx[i]], dim=0)
                x = self.resblocks[i](x)

        return x

class ImageEncoder_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model,add_gate=True, gate = None):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.cfg = cfg
        self.ln_post = clip_model.visual.ln_post
        self.clip_model = clip_model
        # self.proj = clip_model.visual.proj
        self.proj = ProjLearner(clip_model)
        self.transformer = Transformer_VPTD(cfg, classnames, clip_model)
        self.add_gate = add_gate
        if self.add_gate:
            self.gate = gate

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class_embedding is class token.
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])  # only take class token which is awsome.

        x = self.proj(x)

        if self.add_gate:
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
        x = gate * x

        return x

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, visual_gate, text_gate):
        super().__init__()
        self.visual_gate = visual_gate
        self.text_gate = text_gate
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model, gate=self.visual_gate)
        self.text_encoder = TextEncoder(clip_model, gate=self.text_gate)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = CustomAdapter(visual_gate = self.visual_gate, text_gate = self.text_gate).half()
        self.cfg = cfg

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        x, y = self.adapter(image_features, text_features)
        ratio = 0.5


        image_features = ratio * x + (1 - ratio) * image_features
        text_features = ratio * y + (1 - ratio) * text_features
        # image_features = gate1 * x + (1 - gate1) * image_features
        # text_features = gate2 * y + (1 - gate2) * text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

@TRAINER_REGISTRY.register()
class Light_DPALG(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # 创建gate模块
        self.visual_gate = nn.Linear(512, 1).half()
        # self.visual_gate.apply(init_bert_weights)

        self.text_gate = nn.Linear(512, 1).half()
        # self.text_gate.apply(init_bert_weights)

        print("加入Lora模型")
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["c_fc"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )

        clip_lora_model = get_peft_model(clip_model, config)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_lora_model, self.visual_gate, self.text_gate)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt" not in name and "adapter" not in name and "lora" not in name and "gate" not in name:
                param.requires_grad_(False)

        print("gate的模块有：")
        for name, param in self.model.named_parameters():
            if "gate" in name:
                print(name)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            # load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # 计算推理时间
        iterations = 300
        random_input = torch.randn(8, 3, 224, 224).to(self.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # GPU预热
        for _ in range(50):
            _ = self.model(random_input)
            # 获取程序运行期间分配的显存最大值（以字节为单位）
            max_allocated_memory = torch.cuda.max_memory_allocated()
            # 将字节转换为GB
            max_allocated_memory_MB = max_allocated_memory / (1024 ** 3)
            print(f"程序运行期间分配的最大显存: {max_allocated_memory_MB:.2f} GB")

        times = torch.zeros(iterations)
        with torch.no_grad():
            for iter in range(iterations):
                starter.record()
                _ = self.model(random_input)
                ender.record()
                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()
        print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))


        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)


        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        split = "test"

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            label, inputs = self.parse_batch_test(batch)
            output = self.model(inputs)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def reshape_transform(self, tensor, height=14, width=14):
        # 假设输入 tensor 的 shape 为 [197, 1, 768]
        tensor = tensor.permute(1, 0, 2)  # 调整维度至 [1, 197, 768]
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def grad_cam(self):
        # self.set_model_mode("eval")
        # model = self.model

        model = self.clip_model.visual.eval()
        target_layers = [model.transformer.resblocks[-1].ln_1]

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()  # 如果是gpu的话加速
        model = model.float()

        # target_layers = [model.adapter.image_adapter.norm2]
        reshape_transform = self.reshape_transform
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

        img_name = "yachong_1.jpg"
        image_path = "./DATA/gouqi/image_9k/yachong/" + img_name

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))

        # trans =  transforms.Compose([
        #     transforms.RandomResizedCrop(size=(224,224), ratio=(0.75, 1.3333), scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        # ])

        # 预处理图像
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img.astype(dtype=np.float32) / 255, grayscale_cam)
        # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)  ##注意自己图像格式，吐过本身就是BGR，就不用这行

        output_path = "gradcam_out/" + self.cfg.TRAINER.NAME + "/" + "seed"+ str(self.cfg.SEED)
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(output_path + "/" + img_name, visualization)

    def plot_tsne(self, features, labels, out_path, seed):
        '''
        features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
        label:(N) 有N个标签
        '''
        # 清理当前图形，避免数据重叠
        import seaborn as sns
        import pandas as pd
        plt.figure()
        tsne = TSNE(n_components=2, init='pca', random_state=seed, perplexity=30, learning_rate=200, n_iter=1000)
        class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4

        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
        print('tsne_features的shape:', tsne_features.shape)
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化

        df = pd.DataFrame()
        df["y"] = labels
        df["comp-1"] = tsne_features[:, 0]
        df["comp-2"] = tsne_features[:, 1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", class_num),
                        data=df).set(title="T-SNE")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.8)
        plt.savefig(out_path + str(seed) + ".jpg", dpi=800)
        plt.close()

    def t_sne(self):
        dataloader = self.test_loader
        ### 本模型
        model1 = self.model.image_encoder.to("cuda")
        model_adapter = self.model.adapter.image_adapter.float()
        model1.eval()
        model1 = model1.float()
        ### clip原模型
        model2 = self.clip_model.visual
        model2.eval()
        model2 = model2.float()

        features1 = []
        features2 = []
        features3 = []

        labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                label, images = self.parse_batch_test(batch)
                images = images.to("cuda")

                outputs1 = model1(images)
                outputs2 = model_adapter(outputs1)
                outputs3 = model2(images)

                features1.append(outputs1.cpu().numpy())
                features2.append(outputs2.cpu().numpy())
                features3.append(outputs3.cpu().numpy())

                labels.append(label.cpu().numpy())

        features1 = np.concatenate(features1, axis=0)
        features2 = np.concatenate(features2, axis=0)
        features3 = np.concatenate(features3, axis=0)
        print(features2.shape)
        labels = np.concatenate(labels, axis=0)

        print(self.cfg.DATASET.NUM_SHOTS)
        out_path = "TSNE_out/" + self.cfg.DATASET.NAME + "/" + self.cfg.TRAINER.NAME + "/" + str(
            self.cfg.DATASET.NUM_SHOTS) + "/"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        seed = self.cfg.SEED
        self.plot_tsne(features2, labels, out_path, seed)
        # self.plot_tsne(features3, labels, out_path, seed)

    def count_param(self):
        model = self.model
        # 计算参数量
        total_params1 = sum(p.numel() for p in model.parameters())
        total_params2 = 149620737

        rate =  (total_params1 - total_params2) /  total_params2

        million_value = (total_params1 - total_params2) / 1000000

        print(f'Dual (PAL)G Total number of parameters: {total_params1}')
        print(f'CLIP Total number of parameters: {total_params2}')
        print(f"所增加的参数: {million_value:.2f}M")
        print("所增加的参数比例：", rate)

    def parse_batch_test(self, batch):
        if self.cfg.TEST.EVALUATOR == "Image_Text_Retrieval":
            input = batch["img"]
            text = batch["text"]

            input = input.to(self.device)
            return input, text
        else:
            input = batch["img"]
            label = batch["label"]

            input = input.to(self.device)
            label = label.to(self.device)

            return label, input

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
