import ast
import contextlib
import re
import yaml
from pathlib import Path

import torch
import torch.nn as nn

from ..backbone.common import FrozenBatchNorm2d
from ..backbone.hgnetv2 import HG_Stage, StemBlock
from ..core import register
from ..deim.dfine_decoder import DFINETransformer
from ..deim.hybrid_encoder import (
    CSPLayer,
    ConvNormLayer_fuse,
    RepNCSPELAN4,
    SCDown,
    TransformerEncoderBlock,
)
from ..misc.dist_utils import is_dist_available_and_initialized
from .custom_nn.neck.DSPR import CGRF, DSPR
from .ultralytics_nn.conv import Concat


RED, GREEN, BLUE, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[38;5;208m", "\033[0m"


@register(force=True)
class DEIM_MG(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        yaml_path,
        pretrained=None,
        freeze_stem_only=False,
        freeze_at=-1,
        freeze_norm=False,
        num_classes=80,
        eval_spatial_size=(640, 640),
    ):
        super().__init__()
        d = yaml_load(yaml_path)
        backbone, encoder, decoder, self.save = parse_model(
            d,
            ch=3,
            nc=num_classes,
            eval_spatial_size=eval_spatial_size,
            verbose=True,
        )
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        if freeze_at >= 0:
            self._freeze_parameters(self.backbone[0])
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.backbone))):
                    self._freeze_parameters(self.backbone[i])

        if freeze_norm:
            self._freeze_norm(self.backbone)

        if pretrained:
            try:
                state = torch.load(pretrained, map_location="cpu")
                print(f"Loaded stage1 {pretrained} HGNetV2 from local file.")
                print(RED + f"Loading Pretrained State Dict Key Names:{state.keys()}" + RESET)
                self.backbone.load_state_dict(state, strict=False)
            except (Exception, KeyboardInterrupt) as e:
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) \
                        or (not is_dist_available_and_initialized()):
                    print(f"Loading Backbone Pretrained Weight Error. Message:{str(e)}")
                raise

    def forward(self, x, targets=None):
        y = []
        feature_modules = list(self.backbone.children()) + list(self.encoder.children())
        for m in feature_modules:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        x = self.decoder([y[j] for j in self.decoder.f], targets)
        if self.training and targets is not None:
            dga_maps = self._collect_aux_maps(feature_modules, "dga_map")
            if dga_maps:
                x["dga_maps"] = dga_maps
            sgds_maps = self._collect_aux_maps(feature_modules, "sgds_map")
            if sgds_maps:
                x["sgds_maps"] = sgds_maps
        return x

    @staticmethod
    def _collect_aux_maps(modules, attr):
        maps = []
        for module in modules:
            response_map = getattr(module, attr, None)
            if response_map is not None:
                maps.append({"map": response_map})
                setattr(module, attr, None)
        return maps

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False


@register(force=True)
class DRQ_DETR(DEIM_MG):
    __share__ = DEIM_MG.__share__

    def __init__(
        self,
        yaml_path,
        pretrained=None,
        freeze_stem_only=False,
        freeze_at=-1,
        freeze_norm=False,
        num_classes=80,
        eval_spatial_size=(640, 640),
    ):
        super().__init__(
            yaml_path=yaml_path,
            pretrained=pretrained,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            num_classes=num_classes,
            eval_spatial_size=eval_spatial_size,
        )


def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file}"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def parse_module(d, i, f, m, args, ch, nc=None, eval_spatial_size=None):
    if isinstance(m, str):
        m = getattr(torch.nn, m[3:]) if m.startswith("nn.") else globals()[m]

    if isinstance(args, list):
        args = list(args)
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

    c2 = ch[-1]
    if m in {StemBlock, HG_Stage}:
        c1, cmid, c2 = ch[f], args[0], args[1]
        args = [c1, cmid, c2, *args[2:]]
    elif m in {RepNCSPELAN4, CSPLayer, ConvNormLayer_fuse, SCDown}:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m is TransformerEncoderBlock:
        c2 = ch[f]
        args = [c2, *args]
    elif m is Concat:
        c2 = sum(ch[x] for x in f)
    elif m in {DSPR, CGRF}:
        c1 = [ch[i] for i in f] if isinstance(f, list) else ch[f]
        c2 = args[0]
        args = [c1, c2, *args[1:]]
    elif m is DFINETransformer:
        args["feat_channels"] = [ch[x] for x in f]
        args["num_classes"] = nc
        args["eval_spatial_size"] = eval_spatial_size
    else:
        c2 = ch[f]

    m_ = m(**args) if isinstance(args, dict) else m(*args)
    t = str(m)[8:-2].replace("__main__.", "")
    m_.np = sum(x.numel() for x in m_.parameters())
    m_.i, m_.f, m_.type = i, f, t
    return m_, c2, t, args


def parse_model(d, ch, nc, eval_spatial_size, verbose=True):
    if verbose:
        print(ORANGE + f"{'':>3}{'from':>10}{'params':>10}  {'module':<60}{'arguments':<30}" + RESET)

    layer_index, ch = 0, [ch]
    backbone_layers, encoder_layers, decoder_model, save, c2 = [], [], None, [], ch[-1]

    if verbose:
        print(BLUE + "-" * 40 + "BackBone" + "-" * 40 + RESET)
    for f, m, args in d["backbone"]:
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch)
        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)
        backbone_layers.append(m_)
        if layer_index == 0:
            ch = []
        ch.append(c2)
        layer_index += 1

    if verbose:
        print(BLUE + "-" * 40 + "Encoder" + "-" * 40 + RESET)
    for f, m, args in d["encoder"]:
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch)
        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)
        encoder_layers.append(m_)
        ch.append(c2)
        layer_index += 1

    if verbose:
        print(BLUE + "-" * 40 + "Decoder" + "-" * 40 + RESET)
    for f, m, args in d["decoder"]:
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch, nc, eval_spatial_size)
        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)
        decoder_model = m_
        ch.append(c2)

    return nn.Sequential(*backbone_layers), nn.Sequential(*encoder_layers), decoder_model, sorted(save)
