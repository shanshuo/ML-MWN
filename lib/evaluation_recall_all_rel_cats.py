# Adapted from https://github.com/yuweihao/KERN/blob/master/lib/evaluation/sg_eval_all_rel_cates.py
import math
import torch
import torch.nn as nn
import numpy as np
import pickle
from functools import reduce
import sys
from lib.ults.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps

class BasicSceneGraphEvaluator:
    def __init__(self, mode, AG_object_classes, AG_all_predicates, AG_attention_predicates, AG_spatial_predicates, AG_contacting_predicates,
                 iou_threshold=0.5, constraint=False, semithreshold=None):
        self.result_dict = {}
        self.mode = mode
        self.rel_cats = {0: 'looking_at', 1: 'not_looking_at', 2: 'unsure', 3: 'above', 4: 'beneath', 
            5: 'in_front_of', 6: 'behind', 7: 'on_the_side_of', 8: 'in', 9: 'carrying', 
            10: 'covered_by', 11: 'drinking_from', 12: 'eating', 13: 'have_it_on_the_back', 14: 'holding', 
            15: 'leaning_on', 16: 'lying_on', 17: 'not_contacting', 18: 'other_relationship', 19: 'sitting_on', 
            20: 'standing_on', 21: 'touching', 22: 'twisting', 23: 'wearing', 24: 'wiping', 25: 'writing_on',
            26: 'all_rel_cates'}
        
        # in_front_of	not_looking_at	holding	looking_at	not_contacting	on_the_side_of	beneath	touching	behind	sitting_on	unsure	in	leaning_on	other_relationship	standing_on wearing drinking_from   covered_by	above	carrying	eating	lying_on	writing_on	wiping	have_it_on_the_back	twisting
        # self.order = [5, 1, 14, 0, 17, 7, 4, 21, 6, 19, 2, 8, 15, 18, 20, 23, 11, 10, 3, 9, 12, 16, 25, 24, 13, 22, 26]  # train set's relation frequency reverse order 
        self.order = [1, 0, 2, 5, 7, 4, 6, 3, 13, 14, 17, 21, 19, 8, 15, 18, 20, 23, 11, 10, 9, 12, 16, 25, 24, 22]  # decreasly order by sample number, grouped by attention, spatial and contacting types
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        for k, v in self.result_dict[self.mode + '_recall'].items():
            self.result_dict[self.mode + '_recall'][k] = {}
            for rel_cat_id, rel_cat_name in self.rel_cats.items():
                self.result_dict[self.mode + '_recall'][k][rel_cat_name] = []
        self.constraint = constraint # semi constraint if True
        self.iou_threshold = iou_threshold
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        self.semithreshold = semithreshold

    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        for k, v in self.result_dict[self.mode + '_recall'].items():
            self.result_dict[self.mode + '_recall'][k] = {}
            for rel_cat_id, rel_cat_name in self.rel_cats.items():
                self.result_dict[self.mode + '_recall'][k][rel_cat_name] = []

    def print_stats(self):
        outputs = {}  # {k: [mR, R, attention mR, spatial mR, contact mR, attention R, spatial R, contact R], ...}
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            outputs[k] = [None for x in range(31)]
            rel_cat_rec = []  # list of each relationship class's frame-level averaged recall
            relationship_category_names = []  # list of each relationship class's name
            # for rel_cat_id, rel_cat_name in self.rel_cats.items():
            for rel_cat_id in self.order:
                rel_cat_name = self.rel_cats[rel_cat_id]
                if rel_cat_name == 'all_rel_cates':  # this includes all relationship classes, ignore it
                    continue  # this should be put before the following mean
                else:
                    category_recall = np.mean(v[rel_cat_name])
                    relationship_category_names.append(rel_cat_name)
                    rel_cat_rec.append(category_recall)
                    # print(f'R@{k}: {category_recall:.4f}', rel_cat_name)  # verbose
            print(f'mR@{k}: {np.mean(rel_cat_rec):.4f}')
            print(f'R@{k}: {np.mean(v["all_rel_cates"]):.4f}')
            print(f'Per relationship recall@{k}')
            print(', '.join(relationship_category_names))
            print(', '.join('%.4f' % x for x in rel_cat_rec))
            outputs[k][0] = np.mean(rel_cat_rec)  # mR
            outputs[k][1] = np.mean(v["all_rel_cates"])  # R
            outputs[k][2] = np.mean(rel_cat_rec[:3])  # attention mR
            outputs[k][3] = np.mean(rel_cat_rec[3:6])  # spatial mR
            outputs[k][4] = np.mean(rel_cat_rec[9:])  # contact mR
            for i, r in enumerate(rel_cat_rec):
                outputs[k][i+4] = r

        return outputs

    def evaluate_scene_graph(self, gt, pred):
        '''collect the groundtruth and prediction'''

        pred['attention_distribution'] = nn.functional.softmax(pred['attention_distribution'], dim=1)

        for idx, frame_gt in enumerate(gt):
            # generate the ground truth
            gt_boxes = np.zeros([len(frame_gt), 4]) #now there is no person box! we assume that person box index == 0; (frame_gt_box_num, 4)
            gt_classes = np.zeros(len(frame_gt))  # (frame_gt_box_num), frame_gt_box_num = frame_gt_object_box_num + 1(frame_gt_person_box_num)
            gt_relations = []
            human_idx = 0
            gt_classes[human_idx] = 1  # the first item is person
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            for m, n in enumerate(frame_gt[1:]):  # loop this frame's object boxes
                # each pair
                gt_boxes[m+1,:] = n['bbox']  # object bbox
                gt_classes[m+1] = n['class']  # object label id
                gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_attention_predicates[n['attention_relationship']])]) # for attention triplet <human-object-predicate>; [human_box_idx, object_box_idx, attention_label_id]
                #spatial and contacting relationship could be multiple; treat multi-label as multiple samples
                for spatial in n['spatial_relationship'].numpy().tolist():  # [object_box_idx, human_box_idx, spatial_label_id]
                    gt_relations.append([m+1, human_idx, self.AG_all_predicates.index(self.AG_spatial_predicates[spatial])]) # for spatial triplet <object-human-predicate>
                for contact in n['contacting_relationship'].numpy().tolist():  # [human_box_idx, object_box_idx, contact_label_id]
                    gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_contacting_predicates[contact])])  # for contact triplet <human-object-predicate>

            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),  # (frame_sample_num, 3), each sample has only one relationship label, 3: [human/object_box_idx, human/object_box_idx, relationship_label_id]
                'gt_boxes': gt_boxes,
            }

            # first part for attention and contact, second for spatial

            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),             #attention, human-object, (frame_pred_object_num, 2)
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:,::-1],     #spatial, object-human so reverse order
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()), axis=0)    #contacting, human-object
            # (frame_pred_pair_num*3, 2), here frame_pred_pair_num = frame_pred_object_num, for predcls frame_pred_pair_num = frame_gt_pair_num, 3 is for three relationship types, 2 is for human and object index

            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),  # only attention scores, spatial and contact set as 0; (frame_pred_pair_num, 3)
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)  # (frame_pred_pair_num, 26)
            pred_scores_2 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)  # (frame_pred_pair_num, 26)
            pred_scores_3 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)  # (frame_pred_pair_num, 26)

            if self.mode == 'predcls':

                pred_entry = {
                    'pred_boxes': pred['boxes'][:,1:].cpu().clone().numpy(),  # (frame_pred_obj_num, 3)
                    'pred_classes': pred['labels'].cpu().clone().numpy(),  # (frame_pred_obj_num,)
                    'pred_rel_inds': rels_i,  # (frame_pred_pair_num*3, 2)
                    'obj_scores': pred['scores'].cpu().clone().numpy(),  # (frame_pred_obj_num,)
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)  # (frame_pred_pair_num, 26)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }

            evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                               iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semithreshold, rel_cats=self.rel_cats)

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold = 0.9, rel_cats=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    gt_rels_nums = [0 for x in range(len(rel_cats))]
    for rel in gt_rels:
        gt_rels_nums[rel[2]] += 1  # rel[2]: get relationship label id
        gt_rels_nums[-1] += 1  # -1 is the index for all_rel_cats

    pred_rel_inds = pred_entry['pred_rel_inds']  # (frame_pred_pair_num*3, 2)
    rel_scores = pred_entry['rel_scores']


    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']

    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            if rel_scores[i,0]+rel_scores[i,1] > 0:
                # this is the attention distribution
                pred_rels.append(np.append(j,rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            elif rel_scores[i,3]+rel_scores[i,4] > 0:
                # this is the spatial distribution
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])
            elif rel_scores[i,9]+rel_scores[i,10] > 0:
                # this is the contact distribution
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])

        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)  # person_score * object_score, (frame_pred_pair_num*3,)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores  # (frame_pred_pair_num*3, 26)
        score_inds = argsort_desc(overall_scores)[:100]  # (frame_pred_pair_num*3*26, 2)
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))  # (frame_pred_pair_num*3*26, 3)
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]  # (frame_pred_pair_num*3,)

    else:  # with constraint, default
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) # 1+  dont add 1 because no dummy 'no relations'
        predicate_scores = rel_scores.max(1)


    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet', rel_cats=rel_cats,
                **kwargs)

    for k in result_dict[mode + '_recall']:
        for rel_cat_id, rel_cat_name in rel_cats.items():
            if gt_rels_nums[rel_cat_id] == 0:
                continue  # ignore the image that doesn't have this type of relationship
            match = reduce(np.union1d, pred_to_gt[rel_cat_name][:k])
            rec_i = float(len(match)) / float(gt_rels_nums[rel_cat_id])
            result_dict[mode + '_recall'][k][rel_cat_name].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores

###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False, rel_cats=None):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    #assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    sorted_scores = relation_scores.prod(1)  # person_score * predicate_score * object_score = relation_score
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)

    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
        rel_cats=rel_cats,  # https://www.python.org/dev/peps/pep-0008/#when-to-use-trailing-commas
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False, rel_cats=None):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    # pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    pred_to_gt = dict()
    for rel_cat_id, rel_cat_name in rel_cats.items():
        pred_to_gt[rel_cat_name] = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            # pred_to_gt[i].append(int(gt_ind))
            pred_to_gt['all_rel_cates'][i].append(int(gt_ind))
            pred_to_gt[rel_cats[gt_triplets[int(gt_ind), 1]]][i].append(int(gt_ind))
    return pred_to_gt


def calculate_mR_from_evaluator_list(evaluator_list, mode, constraint=None, save_file=None):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        print('\n')
        print('relationship: ', pred_name)
        rel_results = evaluator_rel[mode].print_stats()
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR10 = 0.0
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']):
            continue
        mR10 += value['R@10']
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
    rel_num = len(evaluator_list)
    mR10 /= rel_num
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall['R@10'] = mR10
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall


    # if constraint is None:
    #     recall_mode = ''
    # elif constraint == 'semi':
    #     recall_mode = 'mean recall with semi-constraint'
    # else:  # no constraint
    #     recall_mode = 'mean recall without constraint'
    print('\n')
    print(f'======================{mode}  mean recall with {constraint} constraint============================')
    print(f'mR@10: {mR10}')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)

    # if save_file is not None:
    #     if multiple_preds:
    #         save_file = save_file.replace('.pkl', '_multiple_preds.pkl')
    #     with open(save_file, 'wb') as f:
    #         pickle.dump(all_rel_results, f)

    return mean_recall


def eval_entry(mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    evaluator[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    evaluator_multiple_preds[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    for (pred_id, _, evaluator_rel), (_, _, evaluator_rel_mp) in zip(evaluator_list, evaluator_multiple_preds_list):
        gt_entry_rel = gt_entry.copy()
        mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
        if gt_entry_rel['gt_relations'].shape[0] == 0:
            continue
        
        evaluator_rel[mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
        )
        evaluator_rel_mp[mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
        )

