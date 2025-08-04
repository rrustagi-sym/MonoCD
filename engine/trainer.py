import datetime
import logging
import time
import pdb
import os
import numpy as np

import torch
import torch.distributed as dist

# Fix for distutils.version issue in newer Python versions
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorBoard not available. Error: {e}")
    print("Install tensorboard with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from engine.inference import inference
from utils import comm
from utils.metric_logger import MetricLogger
from utils.metrics import DetectionMetrics, format_metrics_for_tensorboard
from utils.comm import get_world_size
from torch.nn.utils import clip_grad_norm_

def reduce_loss_dict(loss_dict):
	"""
	Reduce the loss dictionary from all processes so that process with rank
	0 has the averaged results. Returns a dict with the same fields as
	loss_dict, after reduction.
	"""
	world_size = get_world_size()
	if world_size < 2:
		return loss_dict
	with torch.no_grad():
		loss_names = []
		all_losses = []
		for k in sorted(loss_dict.keys()):
			loss_names.append(k)
			all_losses.append(loss_dict[k])

		all_losses = torch.stack(all_losses, dim=0)
		dist.reduce(all_losses, dst=0)

		reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

	return reduced_losses

def calculate_validation_loss(cfg, model, data_loaders_val, device):
	"""
	Calculate validation loss on the validation set.
	
	Args:
		cfg: Configuration object
		model: Model to evaluate
		data_loaders_val: Validation data loader
		device: Device to run on
		
	Returns:
		dict: Dictionary containing validation losses
	"""
	model.eval()
	total_losses = {}
	num_batches = 0
	
	logger = logging.getLogger("monocd.trainer")
	logger.info("Calculating validation loss...")
	
	with torch.no_grad():
		for batch in data_loaders_val:
			images = batch["images"].to(device)
			targets = [target.to(device) for target in batch["targets"]]
			
			# Forward pass - need to temporarily set to training mode to get losses
			model.train()
			loss_dict, log_loss_dict = model(images, targets)
			model.eval()
			
			# Accumulate losses
			for key, value in loss_dict.items():
				if key not in total_losses:
					total_losses[key] = 0.0
				total_losses[key] += value.item()
			
			num_batches += 1
	
	# Average losses
	avg_losses = {}
	for key, value in total_losses.items():
		avg_losses[key] = value / num_batches
	
	# Calculate total loss
	total_loss = sum(avg_losses.values())
	avg_losses['total_loss'] = total_loss
	
	return avg_losses


def calculate_detection_metrics_batch(model, batch, device, iou_threshold=0.5):
	"""
	Calculate detection metrics for a single batch.
	
	Args:
		model: Model to evaluate
		batch: Batch of data containing images and targets
		device: Device to run on
		iou_threshold: IoU threshold for considering a detection as correct
		
	Returns:
		dict: Dictionary containing detection metrics
	"""
	model.eval()
	metrics_calculator = DetectionMetrics(iou_threshold=iou_threshold)
	
	with torch.no_grad():
		images = batch["images"].to(device)
		targets = [target.to(device) for target in batch["targets"]]
		
		# Get predictions
		output, eval_utils, _ = model(images, targets)
		
		# Convert predictions and ground truth to format expected by metrics calculator
		predictions = []
		ground_truth = []
		
		# Process each image in the batch
		for i in range(len(targets)):
			target = targets[i]
			pred_boxes = output[i] if output.dim() > 1 else output
			
			# Extract ground truth boxes
			if hasattr(target, 'bbox') and target.bbox is not None:
				gt_boxes = target.bbox.cpu().numpy()
				gt_labels = target.get_field('labels').cpu().numpy() if hasattr(target, 'get_field') else []
				
				for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
					ground_truth.append({
						'bbox': box,
						'class_id': label
					})
			
			# Extract predicted boxes
			if pred_boxes is not None and len(pred_boxes) > 0:
				pred_boxes_np = pred_boxes.cpu().numpy()
				
				# Assuming pred_boxes format is [x1, y1, x2, y2, score, class_id, ...]
				for j in range(len(pred_boxes_np)):
					box = pred_boxes_np[j][:4]  # First 4 elements are bbox coordinates
					score = pred_boxes_np[j][4] if len(pred_boxes_np[j]) > 4 else 1.0
					class_id = pred_boxes_np[j][5] if len(pred_boxes_np[j]) > 5 else 0
					
					predictions.append({
						'bbox': box,
						'class_id': class_id,
						'score': score
					})
		
		# Calculate metrics
		metrics_calculator.update(predictions, ground_truth)
		return metrics_calculator.get_all_metrics()


def calculate_validation_metrics(cfg, model, data_loaders_val, device, iou_threshold=0.5):
	"""
	Calculate validation metrics (precision, recall, accuracy) on the validation set.
	
	Args:
		cfg: Configuration object
		model: Model to evaluate
		data_loaders_val: Validation data loader
		device: Device to run on
		iou_threshold: IoU threshold for considering a detection as correct
		
	Returns:
		dict: Dictionary containing validation metrics
	"""
	model.eval()
	metrics_calculator = DetectionMetrics(iou_threshold=iou_threshold)
	num_batches = 0
	
	logger = logging.getLogger("monocd.trainer")
	logger.info("Calculating validation metrics...")
	
	with torch.no_grad():
		for batch in data_loaders_val:
			images = batch["images"].to(device)
			targets = [target.to(device) for target in batch["targets"]]
			
			# Get predictions
			output, eval_utils, _ = model(images, targets)
			
			# Convert predictions and ground truth to format expected by metrics calculator
			predictions = []
			ground_truth = []
			
			# Check if output is valid
			if output is None or output.numel() == 0:
				logger.warning("Model produced no predictions for this batch, skipping metrics calculation")
				continue
			
			# Convert output to CPU for processing
			output = output.cpu()
			
			# Process each image in the batch
			for i in range(len(targets)):
				target = targets[i]
				pred_boxes = output[i] if output.dim() > 1 else output
				
				# Extract ground truth boxes
				if hasattr(target, 'bbox') and target.bbox is not None:
					gt_boxes = target.bbox.cpu().numpy()
					gt_labels = target.get_field('labels').cpu().numpy() if hasattr(target, 'get_field') else []
					
					for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
						ground_truth.append({
							'bbox': box,
							'class_id': label
						})
				
				# Extract predicted boxes
				if pred_boxes is not None and len(pred_boxes) > 0:
					pred_boxes_np = pred_boxes.cpu().numpy()
					
					# Assuming pred_boxes format is [x1, y1, x2, y2, score, class_id, ...]
					for j in range(len(pred_boxes_np)):
						box = pred_boxes_np[j][:4]  # First 4 elements are bbox coordinates
						score = pred_boxes_np[j][4] if len(pred_boxes_np[j]) > 4 else 1.0
						class_id = pred_boxes_np[j][5] if len(pred_boxes_np[j]) > 5 else 0
						
						predictions.append({
							'bbox': box,
							'class_id': class_id,
							'score': score
						})
			
			# Update metrics
			metrics_calculator.update(predictions, ground_truth)
			num_batches += 1
	
	# Get final metrics
	metrics = metrics_calculator.get_all_metrics()
	
	# Print summary
	metrics_calculator.print_summary()
	
	return metrics

def do_eval(cfg, model, data_loaders_val, iteration):
	eval_types = ("detection",)
	dataset_name = cfg.DATASETS.TEST[0]

	if cfg.OUTPUT_DIR:
		output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
		os.makedirs(output_folder, exist_ok=True)

	evaluate_metric, result_str, dis_ious = inference(
		model,
		data_loaders_val,
		dataset_name=dataset_name,
		eval_types=eval_types,
		device=cfg.MODEL.DEVICE,
		output_folder=output_folder,
	)
	comm.synchronize()

	return evaluate_metric, result_str, dis_ious

def do_train(
		cfg,
		distributed,
		model,
		data_loader,
		data_loaders_val,
		optimizer,
		scheduler,
		warmup_scheduler,
		checkpointer,
		device,
		arguments,
):
	logger = logging.getLogger("monocd.trainer")
	logger.info("Start training")

	meters = MetricLogger(delimiter=" ", )
	max_iter = cfg.SOLVER.MAX_ITERATION
	start_iter = arguments["iteration"]

	# enable warmup
	if cfg.SOLVER.LR_WARMUP:
		assert warmup_scheduler is not None
		warmup_iters = cfg.SOLVER.WARMUP_STEPS
	else:
		warmup_iters = -1

	model.train()

	start_training_time = time.time()
	end = time.time()

	default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH
	grad_norm_clip = cfg.SOLVER.GRAD_NORM_CLIP

	if comm.get_local_rank() == 0:
		if TENSORBOARD_AVAILABLE:
			tensorboard_dir = os.path.join(cfg.OUTPUT_DIR, 'tensorboard_logs')
			os.makedirs(tensorboard_dir, exist_ok=True)
			writer = SummaryWriter(tensorboard_dir)
			logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
			logger.info(f"To view logs, run: tensorboard --logdir {tensorboard_dir}")
		else:
			writer = None
			logger.info("TensorBoard logging disabled (not available)")
		
		best_mAP = 0
		best_result_str = None
		best_iteration = 0
		eval_iteration = 0
		record_metrics = ['Car_bev_', 'Car_3d_']
	
	for data, iteration in zip(data_loader, range(start_iter, max_iter)):
		data_time = time.time() - end

		images = data["images"].to(device)
		targets = [target.to(device) for target in data["targets"]]

		loss_dict, log_loss_dict = model(images, targets)
		losses = sum(loss for loss in loss_dict.values())

		# reduce losses over all GPUs for logging purposes
		log_losses_reduced = sum(loss for key, loss in log_loss_dict.items() if key.find('loss') >= 0)
		meters.update(loss=log_losses_reduced, **log_loss_dict)
		
		optimizer.zero_grad()
		losses.backward()
		
		if grad_norm_clip > 0: clip_grad_norm_(model.parameters(), grad_norm_clip)

		optimizer.step()

		if iteration < warmup_iters:
			warmup_scheduler.step(iteration)
		else:
			scheduler.step(iteration)

		batch_time = time.time() - end
		end = time.time()
		meters.update(time=batch_time, data=data_time)

		iteration += 1
		arguments["iteration"] = iteration

		eta_seconds = meters.time.global_avg * (max_iter - iteration)
		eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

		if comm.get_rank() == 0 and writer is not None:
			depth_errors_dict = {key: meters.meters[key].value for key in meters.meters.keys() if key.find('MAE') >= 0}
			if depth_errors_dict:  # Only log if there are depth errors to log
				writer.add_scalars('train_metric/depth_errors', depth_errors_dict, iteration)
			writer.add_scalar('stat/lr', optimizer.param_groups[0]["lr"], iteration)  # save learning rate

			for name, meter in meters.meters.items():
				if name.find('MAE') >= 0: continue
				if name in ['time', 'data']: 
					writer.add_scalar("stat/{}".format(name), meter.value, iteration)
				else: 
					writer.add_scalar("train_metric/{}".format(name), meter.value, iteration)

		if iteration % 10 == 0 or iteration == max_iter:
			logger.info(
				meters.delimiter.join(
					[
						"eta: {eta}",
						"iter: {iter}",
						"{meters}",
						"lr: {lr:.8f} \n",
					]
				).format(
					eta=eta_string,
					iter=iteration,
					meters=str(meters),
					lr=optimizer.param_groups[0]["lr"],
				)
			)

		if iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0:
			logger.info('iteration = {}, saving checkpoint ...'.format(iteration))
			if comm.get_rank() == 0:
				if 'waymo' in cfg.DATASETS.TEST[0]:
					cur_epoch = iteration // arguments["iter_per_epoch"]
					checkpointer.save("model_checkpoint_{}".format(cur_epoch), **arguments)
				else:
					checkpointer.save("model_checkpoint", **arguments)
			
		if iteration == max_iter and comm.get_rank() == 0:
			checkpointer.save("model_final", **arguments)

		if iteration % cfg.SOLVER.EVAL_INTERVAL == 0:
			if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
				cur_epoch = iteration // arguments["iter_per_epoch"]
				logger.info('epoch = {}, evaluate model on validation set with depth {}'.format(cur_epoch, default_depth_method))
			else:
				logger.info('iteration = {}, evaluate model on validation set with depth {}'.format(iteration, default_depth_method))
			
			# Calculate validation loss
			try:
				val_losses = calculate_validation_loss(cfg, model, data_loaders_val, device)
			except Exception as e:
				logger.warning(f"Error calculating validation loss: {e}")
				val_losses = None
			
			# Calculate detection metrics (precision, recall, accuracy)
			try:
				detection_metrics = calculate_validation_metrics(cfg, model, data_loaders_val, device, iou_threshold=0.5)
			except Exception as e:
				logger.warning(f"Error calculating detection metrics: {e}")
				detection_metrics = None
			
			result_dict, result_str, dis_ious = do_eval(cfg, model, data_loaders_val, iteration)
			
			if comm.get_rank() == 0:
				# Log validation losses to TensorBoard
				try:
					if writer is not None and val_losses is not None:
						for key, value in val_losses.items():
							writer.add_scalar(f"validation_loss/{key}", value, eval_iteration + 1)
						logger.info("Validation Loss: {:.4f}".format(val_losses.get('total_loss', 0.0)))
				except Exception as e:
					logger.warning(f"Error logging validation losses to TensorBoard: {e}")
				
				# Log detection metrics to TensorBoard
				try:
					if writer is not None and detection_metrics is not None:
						tensorboard_metrics = format_metrics_for_tensorboard(detection_metrics)
						for key, value in tensorboard_metrics.items():
							writer.add_scalar(f"detection_metrics/{key}", value, eval_iteration + 1)
						
						# Log macro averages
						precision_values = list(detection_metrics['precision'].values())
						recall_values = list(detection_metrics['recall'].values())
						accuracy_values = list(detection_metrics['accuracy'].values())
						f1_values = list(detection_metrics['f1_score'].values())
						
						if precision_values:
							writer.add_scalar("detection_metrics/macro_precision", np.mean(precision_values), eval_iteration + 1)
							writer.add_scalar("detection_metrics/macro_recall", np.mean(recall_values), eval_iteration + 1)
							writer.add_scalar("detection_metrics/macro_accuracy", np.mean(accuracy_values), eval_iteration + 1)
							writer.add_scalar("detection_metrics/macro_f1", np.mean(f1_values), eval_iteration + 1)
				except Exception as e:
					logger.warning(f"Error logging detection metrics to TensorBoard: {e}")
				
				# only record more accurate R40 results
				result_dict = result_dict[0]
				if len(result_dict) > 0 and writer is not None:
					for key, value in result_dict.items():
						for metric in record_metrics:
							if key.find(metric) >= 0:
								threshold = key[len(metric) : len(metric) + 4]
								writer.add_scalar("eval_{}_{}/{}".format(default_depth_method, threshold, key), float(value), eval_iteration + 1)

				if writer is not None:
					for key, value in dis_ious.items():
						writer.add_scalar("IoUs_{}/{}".format(key, default_depth_method), value, eval_iteration + 1)				

				# record the best model according to the AP_3D, Car, Moderate, IoU=0.7
				important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
				eval_mAP = float(result_dict[important_key])
				if eval_mAP >= best_mAP:
					# save best mAP and corresponding iterations
					best_mAP = eval_mAP
					best_iteration = iteration
					best_result_str = result_str
					checkpointer.save("model_moderate_best_{}".format(default_depth_method), **arguments)

					if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
						logger.info('epoch = {}, best_mAP = {:.2f}, updating best checkpoint for depth {} \n'.format(cur_epoch, eval_mAP, default_depth_method))
					else:
						logger.info('iteration = {}, best_mAP = {:.2f}, updating best checkpoint for depth {} \n'.format(iteration, eval_mAP, default_depth_method))

				eval_iteration += 1
			
			model.train()
			comm.synchronize()

	total_training_time = time.time() - start_training_time
	total_time_str = str(datetime.timedelta(seconds=total_training_time))
	if comm.get_rank() == 0:
		logger.info(
			"Total training time: {} ({:.4f} s / it), best model is achieved at iteration = {}".format(
				total_time_str, total_training_time / (max_iter), best_iteration,
			)
		)

		logger.info('The best performance is as follows')
		if best_result_str is not None:
			logger.info('\n' + best_result_str)
		else:
			logger.info('No evaluation results available')
		
		# Close TensorBoard writer
		if writer is not None:
			writer.close()
