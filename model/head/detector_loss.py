import cv2
import numpy as np
import torch
import math
import torch.distributed as dist
import pdb

from torch.nn import functional as F
from utils.comm import get_world_size

from model.anno_encoder import Anno_Encoder
from model.layers.utils import select_point_of_interest
from model.utils import Uncertainty_Reg_Loss, Laplace_Loss

from model.layers.focal_loss import *
from model.layers.iou_loss import *
from model.head.depth_losses import *
from model.layers.utils import Converter_key2channel

def make_loss_evaluator(cfg):
	loss_evaluator = Loss_Computation(cfg=cfg)
	return loss_evaluator

class Loss_Computation():
	def __init__(self, cfg):
		
		self.anno_encoder = Anno_Encoder(cfg)
		self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
		
		self.max_objs = cfg.DATASETS.MAX_OBJECTS
		self.center_sample = cfg.MODEL.HEAD.CENTER_SAMPLE
		self.regress_area = cfg.MODEL.HEAD.REGRESSION_AREA
		self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE
		self.corner_depth_sp = cfg.MODEL.HEAD.SUPERVISE_CORNER_DEPTH
		self.loss_keys = cfg.MODEL.HEAD.LOSS_NAMES
		self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO

		self.world_size = get_world_size()
		self.dim_weight = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_WEIGHT).view(1, 3)
		self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

		# loss functions
		loss_types = cfg.MODEL.HEAD.LOSS_TYPE
		self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA, cfg.MODEL.HEAD.LOSS_BETA) # penalty-reduced focal loss

		self.horizon_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA, cfg.MODEL.HEAD.LOSS_BETA)

		self.iou_loss = IOULoss(loss_type=loss_types[2]) # iou loss for 2D detection

		# depth loss
		if loss_types[3] == 'berhu': self.depth_loss = Berhu_Loss()
		elif loss_types[3] == 'inv_sig': self.depth_loss = Inverse_Sigmoid_Loss()
		elif loss_types[3] == 'log': self.depth_loss = Log_L1_Loss()
		elif loss_types[3] == 'L1': self.depth_loss = F.l1_loss
		else: raise ValueError

		# regular regression loss
		self.reg_loss = loss_types[1]
		self.reg_loss_fnc = F.l1_loss if loss_types[1] == 'L1' else F.smooth_l1_loss
		self.keypoint_loss_fnc = F.l1_loss

		# multi-bin loss setting for orientation estimation
		self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
		self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
		self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

		self.loss_weights = {}
		for key, weight in zip(cfg.MODEL.HEAD.LOSS_NAMES, cfg.MODEL.HEAD.INIT_LOSS_WEIGHT): self.loss_weights[key] = weight

		# whether to compute corner loss
		self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
		self.compute_corner_loss = 'corner_loss' in self.loss_keys
		self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
		
		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
		self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys
		self.compute_compensated_depth = 'compensated_depth_uncertainty' in self.key2channel.keys
		self.pred_y3d = 'y3d_offset' in self.key2channel.keys
		self.y3d_with_uncertainty = 'y3d_uncertainty' in self.key2channel.keys

		self.pred_ground_plane = cfg.MODEL.HEAD.PRED_GROUND_PLANE
		self.pred_multi_y3d = cfg.MODEL.HEAD.PRED_MULTI_Y3D
		self.train_y3d_kpts_from_gt = cfg.MODEL.HEAD.TRAIN_Y3D_KPTS_FROM_GT
		self.weightincreased = cfg.MODEL.HEAD.WEIGHTINCREASED

		self.use_ground_plane = cfg.USE_GROUND_PLANE

		# new
		self.compute_soft_depth_loss = cfg.COMPUTE_SOFT_DEPTH_LOSS
		self.use_ideal_y3d = cfg.FIXED_Y3D

		self.uncertainty_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT # 1.0
		self.keypoint_xy_weights = cfg.MODEL.HEAD.KEYPOINT_XY_WEIGHT # [1, 1]
		self.keypoint_norm_factor = cfg.MODEL.HEAD.KEYPOINT_NORM_FACTOR # 1.0
		self.modify_invalid_keypoint_depths = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH

		# depth used to compute 8 corners
		self.corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
		self.eps = 1e-5

	def prepare_targets(self, targets):
		# clses
		heatmaps = torch.stack([t.get_field("hm") for t in targets])
		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])
		# 2d detection
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		bboxes = torch.stack([t.get_field("2d_bboxes") for t in targets])
		# 3d detection
		keypoints = torch.stack([t.get_field("keypoints") for t in targets])
		keypoints_depth_mask = torch.stack([t.get_field("keypoints_depth_mask") for t in targets])

		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		EL = torch.stack([t.get_field("EL") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		alphas = torch.stack([t.get_field("alphas") for t in targets])
		orientations = torch.stack([t.get_field("orientations") for t in targets])
		# utils
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
		reg_weight = torch.stack([t.get_field("reg_weight") for t in targets])
		ori_imgs = torch.stack([t.get_field("ori_img") for t in targets])
		trunc_mask = torch.stack([t.get_field("trunc_mask") for t in targets])

		return_dict = dict(cls_ids=cls_ids, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints, dimensions=dimensions,
			locations=locations, EL=EL, rotys=rotys, alphas=alphas, calib=calibs, pad_size=pad_size, reg_mask=reg_mask, reg_weight=reg_weight,
			offset_3D=offset_3D, ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=orientations, keypoints_depth_mask=keypoints_depth_mask,
		)

		if self.use_ground_plane:
			ground_plane = torch.stack([t.get_field("ground_plane") for t in targets])
			return_dict['ground_plane'] = ground_plane

		if self.pred_ground_plane:
			ground_plane = torch.stack([t.get_field("ground_plane") for t in targets])
			horizon_heatmap = torch.stack([t.get_field("horizon_heat_map") for t in targets])
			horizon_state = [t.get_field("horizon_state") for t in targets]
			return_dict['ground_plane'] = ground_plane
			return_dict['horizon_heatmap'] = horizon_heatmap
			return_dict['horizon_state'] = horizon_state

		return heatmaps, return_dict

	def prepare_predictions(self, targets_variables, predictions):
		pred_regression = predictions['reg']
		batch, channel, feat_h, feat_w = pred_regression.shape

		# 1. get the representative points
		targets_bbox_points = targets_variables["target_centers"] # representative points

		reg_mask_gt = targets_variables["reg_mask"]
		flatten_reg_mask_gt = reg_mask_gt.view(-1).bool()

		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1)
		# batch_idxs = batch_idxs[flatten_reg_mask_gt].to(reg_mask_gt.device)
		####### NEW CHANGE ##############
		batch_idxs = batch_idxs.to(flatten_reg_mask_gt.device)
		batch_idxs = batch_idxs[flatten_reg_mask_gt]
		batch_idxs = batch_idxs.to(reg_mask_gt.device) 

		valid_targets_bbox_points = targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt]

		# fcos-style targets for 2D
		target_bboxes_2D = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt]
		target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]
		target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]

		target_regression_2D = torch.cat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points), dim=1)
		mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
		target_regression_2D = target_regression_2D[mask_regression_2D]

		# targets for 3D
		target_clses = targets_variables["cls_ids"].view(-1)[flatten_reg_mask_gt]
		target_depths_3D = targets_variables['locations'][..., -1].view(-1)[flatten_reg_mask_gt]
		target_EL_3D = targets_variables['EL'].view(-1)[flatten_reg_mask_gt]
		target_rotys_3D = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]
		target_alphas_3D = targets_variables['alphas'].view(-1)[flatten_reg_mask_gt]
		target_offset_3D = targets_variables["offset_3D"].view(-1, 2)[flatten_reg_mask_gt]
		target_dimensions_3D = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]
		
		target_orientation_3D = targets_variables['orientations'].view(-1, targets_variables['orientations'].shape[-1])[flatten_reg_mask_gt]
		target_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, target_offset_3D, target_depths_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)

		target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D)
		target_bboxes_3D = torch.cat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]), dim=1)

		target_trunc_mask = targets_variables['trunc_mask'].view(-1)[flatten_reg_mask_gt]
		obj_weights = targets_variables["reg_weight"].view(-1)[flatten_reg_mask_gt]

		# 2. extract corresponding predictions
		pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt]

		pred_regression_2D = F.relu(pred_regression_pois_3D[mask_regression_2D, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois_3D[:, self.key2channel('3d_offset')]

		pred_orientation_3D = torch.cat((pred_regression_pois_3D[:, self.key2channel('ori_cls')], 
									pred_regression_pois_3D[:, self.key2channel('ori_offset')]), dim=1)

		# decode the pred residual dimensions to real dimensions
		pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]
		pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)

		# preparing outputs
		targets = { 'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D, 'orien_3D': target_orientation_3D,
					'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width, 'rotys_3D': target_rotys_3D,
					'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height, 'EL_3D': target_EL_3D,
				}

		preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D, 'dims_3D': pred_dimensions_3D}
		
		reg_nums = {'reg_2D': mask_regression_2D.sum(), 'reg_3D': flatten_reg_mask_gt.sum(), 'reg_obj': flatten_reg_mask_gt.sum()}
		weights = {'object_weights': obj_weights}

		# predict the depth with direct regression
		if self.pred_direct_depth:
			pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D)
			preds['depth_3D'] = pred_direct_depths_3D

		# predict the uncertainty of depth regression
		if self.depth_with_uncertainty:
			preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)
			
			if self.uncertainty_range is not None:
				preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('depth_uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['depth_uncertainty'].mean().item(), preds['depth_uncertainty'].std().item()))

		# predict the keypoints
		if self.compute_keypoint_corner:
			# targets for keypoints
			target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]
			targets['keypoints'] = target_corner_keypoints[..., :2]
			targets['keypoints_mask'] = target_corner_keypoints[..., -1]
			reg_nums['keypoints'] = targets['keypoints_mask'].sum()

			# mask for whether depth should be computed from certain group of keypoints
			target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
			targets['keypoints_depth_mask'] = target_corner_depth_mask

			# predictions for keypoints
			pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]
			pred_keypoints_3D = pred_keypoints_3D.view(flatten_reg_mask_gt.sum(), -1, 2)
			pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D, pred_dimensions_3D,
														targets_variables['calib'], batch_idxs)

			preds['keypoints'] = pred_keypoints_3D			
			preds['keypoints_depths'] = pred_keypoints_depths_3D

		# predict the uncertainties of the solved depths from groups of keypoints
		if self.corner_with_uncertainty:
			preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('corner_uncertainty')]

			if self.uncertainty_range is not None:
				preds['corner_offset_uncertainty'] = torch.clamp(preds['corner_offset_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

			# else:
			# 	print('keypoint depth uncertainty: {:.2f} +/- {:.2f}'.format(
			# 		preds['corner_offset_uncertainty'].mean().item(), preds['corner_offset_uncertainty'].std().item()))

		pred_y3d = None

		if self.pred_y3d:
			pred_y3d_offset = pred_regression_pois_3D[:, self.key2channel('y3d_offset')]
			pred_y3d = self.anno_encoder.decode_y3d(target_clses, pred_y3d_offset)
			preds['y3d'] = pred_y3d

			if self.y3d_with_uncertainty:
				preds['y3d_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('y3d_uncertainty')].squeeze(-1)
				preds['y3d_uncertainty'] = torch.clamp(preds['y3d_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])

		if self.use_ground_plane:
			pred_y3d = self.anno_encoder.decode_y3d_from_ground_plane(targets_variables['ground_plane'], valid_targets_bbox_points,
																	  pred_keypoints_3D, targets_variables['calib'],
																	  targets_variables['pad_size'], batch_idxs)

		if self.pred_ground_plane:
			pred_horizon_heatmaps = predictions['horizon']
			pred_ground_plane = self.anno_encoder.decode_ground_plane_from_heatmap(pred_horizon_heatmaps,
																				   targets_variables['horizon_state'],
																				   targets_variables['calib'],
																				   targets_variables['pad_size'],
																				   )
			if self.train_y3d_kpts_from_gt:
				keypoints = (targets_variables['keypoints'][...,:2] + targets_variables["target_centers"].unsqueeze(2))[reg_mask_gt]
			else:
				keypoints = pred_keypoints_3D
			if self.pred_multi_y3d:
				pred_y3d = self.anno_encoder.decode_multi_y3d_from_ground_plane(pred_ground_plane, valid_targets_bbox_points,
																		  		keypoints, targets_variables['calib'],
																		  		targets_variables['pad_size'], batch_idxs)
			else:
				pred_y3d = self.anno_encoder.decode_y3d_from_ground_plane(pred_ground_plane, valid_targets_bbox_points,
																		  keypoints, targets_variables['calib'],
																		  targets_variables['pad_size'], batch_idxs)

		if self.compute_compensated_depth:
			preds['compensated_depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('compensated_depth_uncertainty')]
			preds['compensated_depth_uncertainty'] = torch.clamp(preds['compensated_depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])
			if pred_y3d is not None:
				if self.pred_multi_y3d:
					compensated_depth = self.anno_encoder.decode_depth_from_roof_and_bottom_multi_y3d(pred_y3d,
																									  valid_targets_bbox_points,
																									  pred_keypoints_3D,
																									  pred_dimensions_3D,
																									  targets_variables['calib'],
																									  targets_variables['pad_size'],
																									  batch_idxs)
				else:
					compensated_depth = self.anno_encoder.decode_depth_from_roof_and_bottom(pred_y3d,
																							valid_targets_bbox_points,
																							pred_keypoints_3D,
																							pred_dimensions_3D,
																							targets_variables['calib'],
																							targets_variables['pad_size'],
																							batch_idxs)
			else:
				if self.use_ideal_y3d:
					# use ideal y3d 1.65m
					ideal_y3d = torch.ones_like(target_EL_3D) * 1.65
					compensated_depth = self.anno_encoder.decode_depth_from_roof_and_bottom(ideal_y3d,
																							valid_targets_bbox_points,
																							pred_keypoints_3D,
																							pred_dimensions_3D,
																							targets_variables['calib'],
																							targets_variables['pad_size'],
																							batch_idxs)
				else:
					# use gt y3d
					compensated_depth = self.anno_encoder.decode_depth_from_roof_and_bottom(target_EL_3D,
																							valid_targets_bbox_points,
																							pred_keypoints_3D,
																							pred_dimensions_3D,
																							targets_variables['calib'],
																							targets_variables['pad_size'],
																							batch_idxs)
			preds['compensated_depths'] = compensated_depth

		# compute the corners of the predicted 3D bounding boxes for the corner loss
		if self.corner_loss_depth == 'direct':
			pred_corner_depth_3D = pred_direct_depths_3D

		elif self.corner_loss_depth == 'keypoint_mean':
			pred_corner_depth_3D = preds['keypoints_depths'].mean(dim=1)

		######################################################
		# new way to compute the corner depth
		elif self.corner_loss_depth == 'key_and_compensated_depth':
			pred_combined_uncertainty = torch.cat(
				[preds['corner_offset_uncertainty'],
				 preds['compensated_depth_uncertainty']], dim=1).exp()
			pred_combined_depths = torch.cat(
				(preds['keypoints_depths'], preds['compensated_depths']), dim=1)
			pred_uncertainty_weights = 1 / pred_combined_uncertainty
			pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
			pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
			preds['weighted_depths'] = pred_corner_depth_3D

		elif self.corner_loss_depth == 'raw_and_compensated_depth':
			pred_combined_uncertainty = torch.cat(
				[preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty'],
				 preds['compensated_depth_uncertainty']], dim=1).exp()
			pred_combined_depths = torch.cat(
				(preds['depth_3D'].unsqueeze(-1), preds['keypoints_depths'], preds['compensated_depths']), dim=1)
			pred_uncertainty_weights = 1 / pred_combined_uncertainty
			pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
			pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
			preds['weighted_depths'] = pred_corner_depth_3D
		##########################################################
		else:
			assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
			# make sure all depths and their uncertainties are predicted
			pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty']), dim=1).exp()
			pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths']), dim=1)
			
			if self.corner_loss_depth == 'soft_combine':
				pred_uncertainty_weights = 1 / pred_combined_uncertainty
				pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
				pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
				preds['weighted_depths'] = pred_corner_depth_3D
			
			elif self.corner_loss_depth == 'hard_combine':
				pred_corner_depth_3D = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		# compute the corners
		pred_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D, pred_corner_depth_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)
		# decode rotys and alphas
		pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, pred_locations_3D)
		# encode corners
		pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D)
		# concatenate all predictions
		pred_bboxes_3D = torch.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]), dim=1)

		preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})

		return targets, preds, reg_nums, weights

	def __call__(self, predictions, targets):
		targets_heatmap, targets_variables = self.prepare_targets(targets)

		if targets_variables['reg_mask'].sum() == 0:
			# no objects in this batch
			return {'all_loss': torch.tensor(0.0, requires_grad=True)}, {'empty_loss': 0.0}

		pred_heatmap = predictions['cls']
		pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)

		# heatmap loss
		if self.heatmap_type == 'centernet':
			hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)
			hm_loss = self.loss_weights['hm_loss'] * hm_loss / torch.clamp(num_hm_pos, 1)

		else: raise ValueError

		if self.pred_ground_plane:
			targets_horizon_heatmap = targets_variables['horizon_heatmap']
			pred_horizon_heatmap = predictions['horizon']
			horizon_hm_loss, num_horizon_pos = self.horizon_loss_fnc(pred_horizon_heatmap, targets_horizon_heatmap)
			UV = pred_horizon_heatmap.shape[-1] * pred_horizon_heatmap.shape[-2]
			# convert UV to tensors
			UV = torch.tensor(UV, dtype=torch.float32, device=pred_horizon_heatmap.device)
			if self.weightincreased:
				horizon_hm_loss = self.loss_weights['horizon_hm_loss'] * horizon_hm_loss / torch.clamp(num_horizon_pos, 1)
			else:
				horizon_hm_loss = self.loss_weights['horizon_hm_loss'] * horizon_hm_loss / torch.clamp(UV, 1)

		# synthesize normal factors
		num_reg_2D = reg_nums['reg_2D']
		num_reg_3D = reg_nums['reg_3D']
		num_reg_obj = reg_nums['reg_obj']
		
		trunc_mask = pred_targets['trunc_mask_3D'].bool()
		num_trunc = trunc_mask.sum()
		num_nontrunc = num_reg_obj - num_trunc

		# IoU loss for 2D detection
		if num_reg_2D > 0:
			reg_2D_loss, iou_2D = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
			reg_2D_loss = self.loss_weights['bbox_loss'] * reg_2D_loss.mean()
			iou_2D = iou_2D.mean()


		if num_reg_3D > 0:
			# direct depth loss
			if self.compute_direct_depth_loss:
				depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'], reduction='none')
				real_depth_3D_loss = depth_3D_loss.detach().mean()
				
				if self.depth_with_uncertainty:
					depth_3D_loss = depth_3D_loss * torch.exp(- preds['depth_uncertainty']) + \
							preds['depth_uncertainty'] * self.loss_weights['depth_loss']
				
				depth_3D_loss = depth_3D_loss.mean()
				depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / (pred_targets['depth_3D'] + self.eps)

			# offset_3D loss
			offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'].cpu(), pred_targets['offset_3D'].cpu(), reduction='none').sum(dim=1)
			offset_3D_loss = offset_3D_loss.to(trunc_mask.device)
			# use different loss functions for inside and outside objects
			if self.separate_trunc_offset:
				if self.trunc_offset_loss_type == 'L1':
					trunc_offset_loss = offset_3D_loss[trunc_mask]
				
				elif self.trunc_offset_loss_type == 'log':
					trunc_offset_loss = torch.log(1 + offset_3D_loss[trunc_mask])

				trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / torch.clamp(trunc_mask.sum(), min=1)
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].mean()
			else:
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.mean()

			# orientation loss
			if self.multibin:
				orien_3D_loss = self.loss_weights['orien_loss'] * \
								Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'], num_bin=self.orien_bin_size)

			# dimension loss
			dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D'], reduction='none') * self.dim_weight.type_as(preds['dims_3D'])
			dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.sum(dim=1).mean()

			if self.pred_y3d:
				y3d_loss = self.loss_weights['y3d_offset_loss'] * self.reg_loss_fnc(preds['y3d'], pred_targets['EL_3D'], reduction='none')
				real_y3d_loss = y3d_loss.detach().mean()
				if self.y3d_with_uncertainty:
					y3d_loss = y3d_loss * torch.exp(- preds['y3d_uncertainty']) + \
								preds['y3d_uncertainty'] * self.loss_weights['y3d_offset_loss']
				y3d_loss = y3d_loss.mean()

			with torch.no_grad():
				try:
					pred_IoU_3D = get_iou_3d(preds['corners_3D'].cpu(), pred_targets['corners_3D'].cpu()).mean()
				except:
					pred_IoU_3D = torch.tensor([0.0]).type_as(preds['corners_3D'])

			# corner loss
			if self.compute_corner_loss:
				# N x 8 x 3
				corner_3D_loss = self.loss_weights['corner_loss'] * \
							self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D'], reduction='none').sum(dim=2).mean()

			if self.compute_compensated_depth:
				pred_compensated_depth, compensated_depth_mask = preds['compensated_depths'], pred_targets['keypoints_depth_mask'].bool()
				target_compensated_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, 3)
				valid_pred_compensated_depth = pred_compensated_depth[compensated_depth_mask]
				invalid_pred_compensated_depth = pred_compensated_depth[~compensated_depth_mask].detach()

				valid_compensated_depth_loss = self.loss_weights['compensated_depth_loss'] * self.reg_loss_fnc(valid_pred_compensated_depth,
								target_compensated_depth[compensated_depth_mask], reduction='none')
				invalid_compensated_depth_loss = self.loss_weights['compensated_depth_loss'] * self.reg_loss_fnc(invalid_pred_compensated_depth,
								target_compensated_depth[~compensated_depth_mask], reduction='none')

				log_compensated_depth_loss = valid_compensated_depth_loss.detach().mean()

				pred_compensated_depth_uncertainty = preds['compensated_depth_uncertainty']
				valid_compensated_depth_uncertainty = pred_compensated_depth_uncertainty[compensated_depth_mask]
				invalid_compensated_depth_uncertainty = pred_compensated_depth_uncertainty[~compensated_depth_mask]

				valid_compensated_depth_loss = valid_compensated_depth_loss * torch.exp(- valid_compensated_depth_uncertainty) + \
												valid_compensated_depth_uncertainty * self.loss_weights['compensated_depth_loss']
				invalid_compensated_depth_loss = invalid_compensated_depth_loss * torch.exp(- invalid_compensated_depth_uncertainty)

				valid_compensated_depth_loss = valid_compensated_depth_loss.sum() / torch.clamp(compensated_depth_mask.sum(), min=1)
				invalid_compensated_depth_loss = invalid_compensated_depth_loss.sum() / torch.clamp(~compensated_depth_mask.sum(), min=1)

				compensated_depth_loss = valid_compensated_depth_loss + invalid_compensated_depth_loss

				compensated_depth_MAE = (preds['compensated_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() / (pred_targets[
					'depth_3D'].unsqueeze(-1)  + self.eps)

				comp_cen_MAE = compensated_depth_MAE[:, 0].mean()
				comp_02_MAE = compensated_depth_MAE[:, 1].mean()
				comp_13_MAE = compensated_depth_MAE[:, 2].mean()


			if self.compute_keypoint_corner:
				# N x 10 x 2
				keypoint_loss = self.loss_weights['keypoint_loss'] * self.keypoint_loss_fnc(preds['keypoints'],
								pred_targets['keypoints'], reduction='none').sum(dim=2) * pred_targets['keypoints_mask']
				
				keypoint_loss = keypoint_loss.sum() / (torch.clamp(pred_targets['keypoints_mask'].sum(), min=1) + self.eps)

				if self.compute_keypoint_depth_loss:
					pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].bool()
					target_keypoints_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, 3)
					
					valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
					invalid_pred_keypoints_depth = pred_keypoints_depth[~keypoints_depth_mask].detach()
					
					# valid and non-valid  针对关键点是否全部露出，分开处理这部分的深度损失
					valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, 
															target_keypoints_depth[keypoints_depth_mask], reduction='none')
					
					invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, 
															target_keypoints_depth[~keypoints_depth_mask], reduction='none')
					
					# for logging
					log_valid_keypoint_depth_loss = valid_keypoint_depth_loss.detach().mean()

					if self.corner_with_uncertainty:
						# center depth, corner 0246 depth, corner 1357 depth
						pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

						valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
						invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]

						valid_keypoint_depth_loss = valid_keypoint_depth_loss * torch.exp(- valid_uncertainty) + \
												self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

						invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * torch.exp(- invalid_uncertainty)

					# average
					valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / torch.clamp(keypoints_depth_mask.sum(), 1)
					invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / torch.clamp((~keypoints_depth_mask).sum(), 1)

					# the gradients of invalid depths are not back-propagated
					if self.modify_invalid_keypoint_depths:
						keypoint_depth_loss = valid_keypoint_depth_loss + invalid_keypoint_depth_loss
					else:
						keypoint_depth_loss = valid_keypoint_depth_loss

					# compute the average error for each method of depth estimation
					keypoint_MAE = (preds['keypoints_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
								   / (pred_targets['depth_3D'].unsqueeze(-1)  + self.eps)

					center_MAE = keypoint_MAE[:, 0].mean()
					keypoint_02_MAE = keypoint_MAE[:, 1].mean()
					keypoint_13_MAE = keypoint_MAE[:, 2].mean()

				# compute the weighted depth loss
				if self.compute_weighted_depth_loss:
					if not self.pred_direct_depth and self.compute_compensated_depth:
						combined_depth = torch.cat(
							(preds['keypoints_depths'], preds['compensated_depths']),
							dim=1)
						combined_uncertainty = torch.cat((preds['corner_offset_uncertainty'],
														  preds['compensated_depth_uncertainty']), dim=1).exp()
						combined_MAE = torch.cat((keypoint_MAE, compensated_depth_MAE), dim=1)

					elif self.compute_compensated_depth:
						combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths'], preds['compensated_depths']), dim=1)
						combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty'], preds['compensated_depth_uncertainty']), dim=1).exp()
						combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE, compensated_depth_MAE), dim=1)
					else:
						combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths']), dim=1)
						combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty']), dim=1).exp()
						combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE), dim=1)

					combined_weights = 1 / combined_uncertainty
					combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
					soft_depths = torch.sum(combined_depth * combined_weights, dim=1)
					if self.compute_soft_depth_loss:
						soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
										  self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'], reduction='mean')

					soft_MAE = ((soft_depths - pred_targets['depth_3D']).abs() / (pred_targets['depth_3D'] + self.eps))
					lower_MAE = torch.min(combined_MAE, dim=1)[0]
					hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
					lower_MAE, hard_MAE, soft_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean()

		loss_dict = {
			'hm_loss':  hm_loss,
			'bbox_loss': reg_2D_loss,
			'dims_loss': dims_3D_loss,
			'orien_loss': orien_3D_loss,
		}

		log_loss_dict = {
			'2D_IoU': iou_2D.item(),
			'3D_IoU': pred_IoU_3D.item(),
		}

		if self.pred_ground_plane:
			loss_dict['horizon_hm_loss'] = horizon_hm_loss

		MAE_dict = {}

		if self.separate_trunc_offset:
			loss_dict['offset_loss'] = offset_3D_loss
			loss_dict['trunc_offset_loss'] = trunc_offset_loss
		else:
			loss_dict['offset_loss'] = offset_3D_loss

		if self.compute_corner_loss:
			loss_dict['corner_loss'] = corner_3D_loss

		if self.pred_direct_depth:
			loss_dict['depth_loss'] = depth_3D_loss
			log_loss_dict['depth_loss'] = real_depth_3D_loss.item()
			MAE_dict['depth_MAE'] = depth_MAE.mean().item()

		if self.pred_y3d:
			loss_dict['y3d_loss'] = y3d_loss
			log_loss_dict['y3d_loss'] = real_y3d_loss.item()

		if self.compute_compensated_depth:
			loss_dict['compensated_depth_loss'] = compensated_depth_loss
			log_loss_dict['compensated_depth_loss'] = log_compensated_depth_loss.item()
			MAE_dict.update({
				'comp_cen_MAE': comp_cen_MAE.item(),
				'comp_02_MAE': comp_02_MAE.item(),
				'comp_13_MAE': comp_13_MAE.item(),
			})

		if self.compute_keypoint_corner:
			loss_dict['keypoint_loss'] = keypoint_loss

			if self.corner_with_uncertainty:
				MAE_dict.update({
					'center_MAE': center_MAE.item(),
					'02_MAE': keypoint_02_MAE.item(),
					'13_MAE': keypoint_13_MAE.item(),
				})

		if self.compute_keypoint_depth_loss:
			loss_dict['keypoint_depth_loss'] = keypoint_depth_loss
			log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss.item()

		if self.compute_weighted_depth_loss:
			if self.compute_soft_depth_loss:
				loss_dict['weighted_avg_depth_loss'] = soft_depth_loss
			MAE_dict.update({
				'lower_MAE': lower_MAE.item(),
				'hard_MAE': hard_MAE.item(),
				'soft_MAE': soft_MAE.item(),
			})

		# loss_dict ===> log_loss_dict
		for key, value in loss_dict.items():
			if not torch.isfinite(value).all():
				print(f"⚠️ NaN/Inf detected in loss: {key}, value: {value}")
				import pdb; pdb.set_trace()
			log_loss_dict[key] = value.item()

		# stop when the loss has NaN or Inf
		for v in loss_dict.values():
			if torch.isnan(v).sum() > 0:
				pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				pdb.set_trace()

		log_loss_dict.update(MAE_dict)

		return loss_dict, log_loss_dict

def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
	gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst

	cls_losses = 0
	reg_losses = 0
	reg_cnt = 0
	for i in range(num_bin):
		# bin cls loss
		cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
		# regression loss
		valid_mask_i = (gt_ori[:, i] == 1)
		cls_losses += cls_ce_loss.mean()
		if valid_mask_i.sum() > 0:
			s = num_bin * 2 + i * 2
			e = s + 2
			pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
			reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
						F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

			reg_losses += reg_loss.sum()
			reg_cnt += valid_mask_i.sum()

	return cls_losses / num_bin + reg_losses / reg_cnt
