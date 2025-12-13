import copy
import json
import time
import os

import torch
import torch.nn as nn

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from . import builder
from utils import dist_utils
from utils.averagemeter import AverageMeter
from utils.logger import *
from utils.metrics import Metrics


def run_trainer(cfg, train_writer=None, val_writer=None):
    """
    Main training function that handles the complete training and validation cycle.

    Args:
        cfg: Configuration object containing all training parameters
        train_writer: TensorBoard writer for training metrics
        val_writer: TensorBoard writer for validation metrics

    Returns:
        None
    """
    logger = get_logger(cfg.log_name)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Build datasets for training and testing
    train_config = copy.deepcopy(cfg.dataset)
    train_config.subset = "train"
    test_config = copy.deepcopy(cfg.dataset)
    test_config.subset = "test"
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(cfg, train_config, logger=logger), \
        builder.dataset_builder(cfg, test_config, logger=logger)
        
    # Build and initialize the model
    base_model = builder.model_builder(cfg.model)
    if cfg.use_gpu:
        base_model.to(local_rank)

    # Initialize training parameters
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Load checkpoints if resuming training or starting from pretrained model
    if cfg.resume_last:
        start_epoch, best_metrics = builder.resume_model(base_model, cfg, logger=logger)
    elif cfg.resume_from is not None:
        builder.load_model(base_model, cfg.resume_from, logger=logger)

    # Print model information for debugging
    if cfg.debug:
        print_log('Trainable_parameters:', logger=logger)
        print_log('=' * 25, logger=logger)
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger=logger)

        print_log('Untrainable_parameters:', logger=logger)
        print_log('=' * 25, logger=logger)
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger=logger)

    # Set up distributed training if needed
    if cfg.distributed:
        if cfg.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
        
    # Set up optimizer and learning rate scheduler
    optimizer = builder.build_optimizer(base_model, cfg)

    if cfg.resume_last:
        builder.resume_optimizer(optimizer, cfg, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, cfg, last_epoch=start_epoch - 1)    
    
    # Initialize Chamfer Distance metrics
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    # Main training loop
    base_model.zero_grad()
    for epoch in range(start_epoch, cfg.max_epoch + 1):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()
        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            dataset_name = cfg.dataset.name
            if dataset_name == "PCN":
                partial = data[0].cuda()
                gt = data[1].cuda()
                
            num_iter += 1
            ret = base_model(partial)
            
            # Calculate losses and backpropagate
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
            _loss = sparse_loss + dense_loss 
            _loss.backward()

            # Update weights after accumulating gradients
            if num_iter == cfg.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(cfg, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            # Process and log loss metrics
            if cfg.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, cfg)
                dense_loss = dist_utils.reduce_tensor(dense_loss, cfg)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            if cfg.distributed:
                torch.cuda.synchronize()

            # Log metrics to TensorBoard
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
            
            # Update timing information
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # Print progress information
            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, cfg.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            # Handle special case for GradualWarmup scheduler
            if cfg.scheduler.type == 'GradualWarmup':
                if n_itr < cfg.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        # Step the learning rate scheduler after each epoch
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        # Log epoch-level training metrics
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        # Run validation at specified frequency
        if epoch % cfg.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, cfg, logger=logger)

            # Save checkpoint if current model is the best so far
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', cfg, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', cfg, logger = logger)      
        if (cfg.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', cfg, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, cfg, logger = None):
    """
    Validate the model on the test dataset.
    
    Args:
        base_model: The model to be validated.
        test_dataloader: DataLoader for the test dataset.
        epoch: Current epoch number.
        ChamferDisL1: Chamfer Distance L1 metric.
        ChamferDisL2: Chamfer Distance L2 metric.
        val_writer: TensorBoard writer for validation metrics.
        cfg: Configuration object.
        logger: Logger for logging messages.
        
    Returns:
        metrics: Computed metrics after validation.
    """
    
    base_model.eval()  

    # Initialize metrics tracking
    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) 

    interval =  n_samples // 10

    # Validation loop
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            dataset_name = cfg.dataset.name
            if dataset_name == "PCN":
                partial = data[0].cuda()
                gt = data[1].cuda()
                if cfg.dataset.mode == "sim3":
                    scale = data[2].cuda()
            
            # Forward pass and loss computation
            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]
            gt = gt
            if cfg.dataset.mode == "sim3":
                coarse_points = coarse_points / scale
                dense_points = dense_points / scale
                gt = gt / scale

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if cfg.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, cfg)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, cfg)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, cfg)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, cfg)
            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt)
            if cfg.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, cfg).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)


            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if cfg.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(cfg.consider_metric, test_metrics.avg())


def run_tester(cfg):
    """
    Test function for evaluating a pre-trained model on the test dataset.
    
    Args:
        cfg: Configuration object containing test parameters including:
            - log_name: Name for the logger
            - dataset.test: Test dataset configuration
            - model: Model architecture configuration
            - ckpts: Path to model checkpoint file
            - use_gpu: Whether to use GPU for inference
            - local_rank: Local rank for GPU selection
            - distributed: Whether to use distributed testing (not implemented)
    
    Returns:
        None
    """
    # Initialize logger for test process
    logger = get_logger(cfg.log_name)
    print_log('Tester start ... ', logger = logger)
    
    # Build test dataset and dataloader
    test_config = copy.deepcopy(cfg.dataset)
    test_config.subset = "test"
    _, test_dataloader = builder.dataset_builder(cfg, test_config)
 
    # Build model architecture
    base_model = builder.model_builder(cfg.model)
    
    # Load pre-trained model weights from checkpoint
    builder.load_model(base_model, cfg.evaluate.checkpoint_path, logger = logger)
    
    # Move model to GPU if specified
    local_rank = int(os.environ["LOCAL_RANK"])
    if cfg.use_gpu:
        base_model.to(local_rank)
    
    # Handle distributed testing (currently not implemented)
    if cfg.distributed:
        raise NotImplementedError()

    # Initialize Chamfer Distance loss functions for evaluation metrics
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    # Run the actual testing process
    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, cfg, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, cfg, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            dataset_name = cfg.dataset.name
            partial = data[0].cuda()
            gt = data[1].cuda()
            scale = data[2].cuda()

            ret = base_model(partial)
            if cfg.dataset.mode == "sim3":
                coarse_points = ret[0]/scale
                dense_points = ret[1]/scale
                gt = gt/scale

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt, require_emd=True)
            # test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)


            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 