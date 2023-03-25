import importlib
import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Tensor

from .decoder import Aux_Module, CosProto_Module


class ModelBuilder(nn.Cell):
    def __init__(self, net_cfg, machine_name):
        super(ModelBuilder, self).__init__()
        self.machine_name = machine_name
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

        self.proto_head = True if net_cfg.get("proto_head", False) else False
        if self.proto_head:
            cfg_rnet = net_cfg["proto_head"]
            self.loss_weight_rnet = cfg_rnet["loss_weight"]
            self.patch_select = cfg_rnet["patch_select"]
            self.select_granularity = cfg_rnet['select_granularity']
            self.num_micro_proto = cfg_rnet['num_micro_proto']
            self.proto_net = CosProto_Module(
                cfg_rnet['in_planes'],
                self._num_classes,
                cfg_rnet['num_micro_proto'],
                cfg_rnet['init_proto_path'],
                cfg_rnet['proto_unpdate_momentum'],
            )
            
    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        enc_cfg["kwargs"].update({"machine_name": self.machine_name})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    
    def construct(self, x, select_mask=None, cutout_mask=None, eval=False, output_proto=False):
        if self._use_auxloss:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x)
                feat = [f1, f2, feat1, feat2]
                outs = self.decoder(feat)
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat[2])
            outs.update({"aux": pred_aux})
        else:
            feat = self.encoder(x)
            outs = self.decoder(feat)
            
        if self.proto_head and output_proto:
            if self.patch_select:
                _, _, h, w = x.shape
                outs["rep_clf"] = ops.ResizeBilinear((h, w), align_corners=True)(outs["rep_clf"])
                if eval:
                    # evaluation phase do not need to do sampling
                    select_mask = ops.ones((h, w), mindspore.float32)
                    pred_proto, _ = self.proto_net(outs["rep_clf"], select_mask)
                else:
                    # TODO: training code transfer to mindspore
                    # if select_mask is None:
                    #     select_mask = self.mask_generation(h, w, self.select_granularity, x.device, cutout_mask)
                    # outs.update({"select_mask": select_mask})
                    # pred_proto, proto_match_idx = self.proto_net(outs["rep_clf"], select_mask)
                    # outs.update({"proto_match_idx": proto_match_idx})
                    pass
            else:
                pred_proto = self.proto_net(outs["rep_clf"])
            outs.update({"proto": pred_proto})
        
        return outs
    
    @staticmethod
    def mask_generation(h, w, select_granularity, device, cutout_mask=None):
        # TODO: training code transfer to mindspore
        # cutout_mask: since cutout_ratio is 2 which cutout half of the image, we do not want the protoclf focus on the cutout part too much
        # assert h == w, 'h and w should be equal'
        # select_mask = torch.zeros((h, w))
        # if cutout_mask is None:
        #     cutout_mask = torch.ones((h, w)).to(device)
        # patch_size = h // select_granularity
        # r_idx = []
        # l_idx = []
        # base_idx = torch.arange(select_granularity) * patch_size
        # for _gran_idx in range(select_granularity):
        #     _r_idx = torch.randint(patch_size, (select_granularity,))
        #     r_idx.append(base_idx + _r_idx)
            
        #     _l_idx = torch.randint(patch_size, (select_granularity,)) + _gran_idx * patch_size
        #     l_idx.append(_l_idx)
        # r_idx_new = torch.cat(r_idx, 0)
        # l_idx_new = torch.cat(l_idx, 0)
        # select_mask[r_idx_new, l_idx_new] = 1
        # select_mask = ((select_mask.to(device) + cutout_mask) == 2).float()
        # return select_mask
        pass

    def update_proto(self, rep, select_mask, proto_match_idx, target, cond):
        # TODO: training code transfer to mindspore
        # rep_selected = rep.permute(0, 2, 3, 1)[:, select_mask == 1]
        # bs, hw, c = rep_selected.shape
        # rep_selected = rep_selected.reshape(bs*hw, c)
        # target_selected = target[:, select_mask == 1].view(-1)
        # valid_mask = target_selected != 255
        # rep_selected_valid = rep_selected[valid_mask]
        # target_selected_valid = target_selected[valid_mask]
        # proto_match_idx_valid = proto_match_idx[valid_mask]
        # cond_valid = cond[valid_mask]
        # proto_match_idx_valid_target = proto_match_idx_valid[torch.arange(len(target_selected_valid)), target_selected_valid]

        # rep_selected_valid_cond = rep_selected_valid[cond_valid]
        # proto_match_idx_valid_target_cond = proto_match_idx_valid_target[cond_valid]
        # target_selected_valid_cond = target_selected_valid[cond_valid]

        # if cond_valid.sum() > 0:
        #     shot_cls = target_selected_valid_cond.unique()
        #     for _cls in shot_cls:
        #         for proto_idx in range(self.num_micro_proto):
        #             _candidate_mask = torch.logical_and(target_selected_valid_cond == _cls, proto_match_idx_valid_target_cond == proto_idx)
        #             if _candidate_mask.sum() > 0:
        #                 self.relation_net.update_proto(rep_selected_valid_cond[_candidate_mask].mean(0), _cls, proto_idx)
        pass
