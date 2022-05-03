import torch
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))
    

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch, exp_name):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2' and data_name == 'scannet':
        from data.scannetv2_inst import Dataset
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
        valid_class_ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

    elif cfg.dataset == "multiscan_inst" and data_name == 'multiscan_inst':
        from data.multiscan_inst import Dataset
        semantic_label_idx = range(1, 21)
        class_labels = ('door', 'table', 'chair', 'cabinet', 'window', 'sofa', 'microwave', 'pillow',
'tv_monitor', 'curtain', 'trash_can', 'suitcase', 'sink', 'backpack', 'bed', 'refrigerator','toilet')
        valid_class_ids = np.array(range(4, 21))
    elif cfg.dataset == "multiscan_part" and data_name == 'multiscan_part':
        from data.multiscan_inst import Dataset
        semantic_label_idx = range(1, 6)
        class_labels = ('static_part', 'door', 'drawer', 'window', 'lid')
        valid_class_ids = np.array(range(1, 6))
    else:
        print("Error: no data loader - " + data_name)
        exit(0)

    dataset = Dataset(test=True)
    dataset.testLoader()
    id2label = {}
    label2id = {}
    for i in range(len(valid_class_ids)):
        label2id[class_labels[i]] = valid_class_ids[i]
        id2label[valid_class_ids[i]] = class_labels[i]
            
        
    dataloader = dataset.test_data_loader

    with torch.no_grad():
        model = model.eval()

        total_end1 = 0.
        matches = {}
        for i, batch in enumerate(dataloader):

            # inference
            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1

            # decode results for evaluation
            N = batch['feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1].split(cfg.filename_suffix)[0]
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            pt_offsets = preds['pt_offsets']    # (N, 3), float32, cuda
            if (epoch > cfg.prepare_epochs):
                scores = preds['score']   # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_idx, proposals_offset, mask_scores = preds['proposals']
                # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) 
                # (nProposal, N), int, cuda
                
                # outlier filtering
                test_mask_score_thre = getattr(cfg, 'test_mask_score_thre', -0.5)
                _mask = mask_scores.squeeze(1) > test_mask_score_thre
                proposals_pred[proposals_idx[_mask][:, 0].long(), proposals_idx[_mask][:, 1].long()] = 1

                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device) \
                    [semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long
                # semantic_id_idx = semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]
 
                # score threshold
                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]
                # semantic_id_idx = semantic_id_idx[score_mask]

                # npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum >= cfg.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]


                # nms (no need)
                if getattr(cfg, 'using_NMS', False):
                    if semantic_id.shape[0] == 0:
                        pick_idxs = np.empty(0)
                    else:
                        proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                        intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                        proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                        pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  
                        # int, (nCluster, N)
                    clusters = proposals_pred[pick_idxs]
                    cluster_scores = scores_pred[pick_idxs]
                    cluster_semantic_id = semantic_id[pick_idxs]
                else:
                    clusters = proposals_pred
                    cluster_scores = scores_pred
                    cluster_semantic_id = semantic_id

                nclusters = clusters.shape[0]


                # prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file, class_labels, valid_class_ids, id2label)

                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt
                
                    if cfg.split == 'val':
                        matches[test_scene_name]['seg_gt'] = batch['labels']
                        matches[test_scene_name]['seg_pred'] = semantic_pred
    

            # save files
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format( \
                        test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()


            logger.info("instance iter: {}/{} point_num: {} ncluster: {} inference time: {:.2f}s".format( \
                batch['id'][0] + 1, len(dataset.test_files), N, nclusters, end1))
            total_end1 += end1

        # evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches, class_labels)
            avgs = eval.compute_averages(ap_scores, class_labels)
            eval.print_results(avgs, class_labels, exp_name)

        logger.info("whole set inference time: {:.2f}s, latency per frame: {:.2f}ms".format(total_end1, total_end1 / len(dataloader) * 1000))

        # evaluate semantic segmantation accuracy and mIoU
        if cfg.split == 'val':
            seg_accuracy = evaluate_semantic_segmantation_accuracy(matches, cfg.ignore_label)
            logger.info("semantic_segmantation_accuracy: {:.4f}".format(seg_accuracy))
            miou = evaluate_semantic_segmantation_miou(matches, cfg.ignore_label)
            logger.info("semantic_segmantation_mIoU: {:.4f}".format(miou))

def evaluate_semantic_segmantation_accuracy(matches, ignore_label):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != ignore_label] == seg_pred_all[seg_gt_all != ignore_label]).sum()
    whole = (seg_gt_all != ignore_label).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches, ignore_label):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    for _index in seg_gt_all.unique():
        if _index != ignore_label:
            intersection = ((seg_gt_all == _index) &  (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            iou_list.append(iou)
    iou_tensor = torch.tensor(iou_list)
    miou = iou_tensor.mean()
    return miou

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()

    exp_name = cfg.config.split('/')[-1][:-5]
    tmp_split = exp_name.split('_')
    model_name = tmp_split[0]
    data_name = '_'.join(tmp_split[2:])

    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'hais':
        from model.hais.hais import HAIS as Network
        from model.hais.hais import model_fn_decorator
    
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))
    model_fn = model_fn_decorator(test=True)

    # load model
    utils.checkpoint_restore(cfg, model, None, cfg.exp_path, cfg.config.split('/')[-1][:-5], 
        use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      
    # resume from the latest epoch, or specify the epoch to restore

    exp_name = cfg.pretrain.split("/")[-2]
    # evaluate
    test(model, model_fn, data_name, cfg.test_epoch, exp_name)