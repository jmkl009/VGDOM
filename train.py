import numpy as np
from time import time
import torch
from datasets import GraphConstuctor
from utils import print_and_log
from tqdm import tqdm
import torch.nn.functional as F



def match(logits, labels, match_label):
    pred_idx = logits[:, match_label].argmax()
    match_label_indices = (labels == match_label).nonzero()
    if len(match_label_indices) == 0:
        return False
    return (pred_idx == match_label_indices[0][0]).item()

class_names = ['Author', 'Title', 'Image']

def print_pti_acc_stats(correct_nums, total_num, log_file, split_name='TRAIN', acc_name='Acc', acc_modifier='top-1'):
    class_accs = np.array(correct_nums)/total_num * 100
    print_and_log('[%s] Avg_class_Accuracy: %.2f%%' % (split_name, class_accs.mean()), log_file)
    for c in range(3):
        print_and_log('%s %s-%s: %.2f%%' % (class_names[c], acc_modifier, acc_name, class_accs[c]), log_file)
    print_and_log('', log_file)

def tree_squash_probs(tree, outputs, leaf_num, device):
    stack = [tree]
    probs = F.sigmoid(outputs)
    # print(probs.shape, probs)
    # probs = outputs
    prob_stack = [probs[tree.idx]]
    squshed_probs = torch.zeros((leaf_num, 4)).to(device)
    while len(stack) > 0:
        curr = stack.pop()
        curr_prob = prob_stack.pop()
        if curr.num_children == 0:
            squshed_probs[curr.idx] = curr_prob
        else:
            stack += curr.children
            prob_stack += [curr_prob * probs[child.idx] for child in curr.children]
    return squshed_probs

def tree_match(tree, probs, true_labels, target_label_pos):
    # print('recurse')
    if tree.num_children > 0:
        max_likelihood = probs[tree.children[0].idx][target_label_pos].item()
        max_idx = 0
        for i, child in enumerate(tree.children):
            if (probs[child.idx][target_label_pos] > max_likelihood).item():
                max_likelihood = probs[child.idx][target_label_pos].item()
                max_idx = i
        return tree_match(tree.children[max_idx], probs, true_labels, target_label_pos)
    return (tree.idx == (true_labels == target_label_pos).nonzero()[0][0]).item()

def convert_labels(labels, leaf_num):
    return torch.argmax(labels[:leaf_num], dim=1)

def train_vgdom(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0

    graph_constructor = GraphConstuctor(model.num_candidates, device=device, model=model)
    best_gcn_eval_acc = 0.0
    save_file_delimited = model_save_file.split('/')
    prev_path = "/".join(save_file_delimited[:-1])
    save_file_delimited = save_file_delimited[-1].split('.')
    gcn_model_save_file = (
            prev_path + "/" + save_file_delimited[0] + '-gcn.' + save_file_delimited[1]
            if len(save_file_delimited) > 1 else 
            prev_path + "/" + save_file_delimited[0] + '-gcn' 
            )
    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, images, leaf_bboxes, extended_bboxes, texts, chars, leaf_tags, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            
            if model.splitted:
                labels = labels.to('cuda:1')
            else:
                labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += leaf_bboxes.shape[0]
            
            optimizer.zero_grad()
            
            encoded_features = model.encode(
                    images.to(device), 
                    leaf_bboxes.to(device), 
                    extended_bboxes.to(device), 
                    texts, chars, leaf_tags.to(device), tag_paths, trees, 
                    rel_bboxes.to(device))
            output = model.decode(encoded_features)
            
            num_bboxes = model.num_bboxes_per_img(leaf_bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            assert len(output) == sum(leaf_nums)

            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs].detach()
                labels_segment = labels[curr_idx:curr_idx + num_leafs].detach()
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))            

                curr_idx += num_leafs

            loss = criterion(output, labels)
            
            graph_losses = []
            curr_idx = 0
            bbox_idx = 0
            for num_leafs, num_bbox, norm_factor in zip(leaf_nums, num_bboxes, norm_factors):
                graph_bboxes = leaf_bboxes[bbox_idx:bbox_idx + num_leafs][:, 1:]
                node_features = encoded_features[curr_idx:curr_idx + num_leafs].detach()
                raw_predicted_prob = output[curr_idx:curr_idx + num_leafs].detach()
                graph_labels = labels[curr_idx:curr_idx + num_leafs]
                graph_and_tmp_labels_and_select_indices = graph_constructor.construct(
                    graph_bboxes, 
                    raw_predicted_prob, 
                    graph_labels, train=True, node_features=node_features, 
                    return_select_indices=True)
                if graph_and_tmp_labels_and_select_indices is not None:
                    graph, tmp_labels, select_indices = graph_and_tmp_labels_and_select_indices
                    logits = model.gcn_forward(graph, node_features[select_indices])
                    graph_loss = criterion(logits, tmp_labels) # F.cross_entropy(logits, tmp_labels)
                    graph_losses.append(graph_loss) # * norm_factor)
                    nc = model.num_candidates
                    (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))


                    gcn_price_correct_macro += pc * norm_factor
                    gcn_title_correct_macro += tc * norm_factor
                    gcn_image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                bbox_idx += num_bbox

            if len(graph_losses) > 0:
                # loss = sum(graph_losses) * (1/len(graph_losses))
                # epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
                # loss.backward()
                # optimizer.step()
                # del loss
                total_graph_loss = sum(graph_losses) * (1/len(graph_losses))  #* len(eval_loader.dataset) / total_domain_num
                loss += total_graph_loss

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            gcn_class_acc = evaluate_vgdom(model, eval_loader, device, 1, 'VAL', log_file)
            gcn_eval_acc = gcn_class_acc.mean()
            if gcn_eval_acc > best_gcn_eval_acc:
                print('GCN Model Saved!', gcn_eval_acc, '>', best_gcn_eval_acc)
                best_gcn_eval_acc = gcn_eval_acc
                torch.save(model.state_dict(), gcn_model_save_file)
            model.train()
        scheduler.step()

    return best_eval_acc, gcn_model_save_file


@torch.no_grad()
def evaluate_vgdom(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()

    graph_constructor = GraphConstuctor(model.num_candidates, device=device, model=model)

    gcn_macro_acc = np.array([0.0, 0.0, 0.0])
    
    site_freq_xpath_dict = dict()
    label_dict = dict()
    tag_path_dict = dict()
    norm_factor_dict = dict()
    output_dict = dict()
    
    total_domain_num = eval_loader.dataset.total_domain_num
    pkl2domain = eval_loader.dataset.pkl2domain
        
    for i, (page_ids, images, leaf_bboxes, extended_bboxes, texts, chars, leaf_tags, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(eval_loader)):
        
        labels_list = labels.tolist()
        label_indices = []
        label_idx = 0
        for num in leaf_nums:
            label_indices.append([])
            label_seg = labels_list[label_idx:label_idx+num]
            label_idx += num
            for j,l in enumerate(label_seg):
                if l == 1:
                    label_indices[-1].append(j)
            for j,l in enumerate(label_seg):
                if l == 2:
                    label_indices[-1].append(j)
            for j,l in enumerate(label_seg):
                if l == 3:
                    label_indices[-1].append(j)
        for batch_id, label_index in enumerate(label_indices):
            page_id = page_ids[batch_id]
            label_dict[page_id] = label_index
        for batch_id, norm_factor in enumerate(norm_factors):
            page_id = page_ids[batch_id]
            norm_factor_dict[page_id] = norm_factor
        tag_path_idx = 0
        tag_paths_segs = []
        for batch_id, num in enumerate(leaf_nums):
            page_id = page_ids[batch_id]
            tag_seg = tag_paths[tag_path_idx:tag_path_idx+num]
            tag_path_idx += num
            tag_path_dict[page_id] = tag_seg
            tag_paths_segs.append(tag_seg)

                    
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]

        encoded_features = model.encode(images.to(device), leaf_bboxes.to(device), extended_bboxes.to(device), texts, chars, leaf_tags.to(device), tag_paths, trees, rel_bboxes.to(device))
        output = model.decode(encoded_features)

        num_bboxes = model.num_bboxes_per_img(leaf_bboxes[:, 0])

        curr_idx = 0
        bbox_idx = 0
        batch_output = []
        for j, (num_leafs, num_bbox, norm_factor) in enumerate(zip(leaf_nums, num_bboxes, norm_factors)):

            graph_bboxes = leaf_bboxes[bbox_idx:bbox_idx + num_leafs][:, 1:]
            graph_labels = labels[curr_idx:curr_idx + num_leafs]
            raw_predicted_prob = output[curr_idx:curr_idx + num_leafs]
            node_features = encoded_features[curr_idx:curr_idx + num_leafs]
            graph, tmp_labels, select_indices = graph_constructor.construct(
                graph_bboxes, raw_predicted_prob, graph_labels, train=False, node_features=node_features, 
                return_select_indices=True)
            logits = model.gcn_forward(graph, node_features[select_indices])

            nc = model.num_candidates
            
            (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))

            gcn_macro_acc[0] += norm_factor * pc
            gcn_macro_acc[1] += norm_factor * tc
            gcn_macro_acc[2] += norm_factor * ic

            
            pred_idx_1 = select_indices[logits[:nc, 1].argmax()]
            pred_idx_2 = select_indices[logits[nc:2*nc, 2].argmax() + nc]
            pred_idx_3 = select_indices[logits[2*nc:3*nc, 3].argmax() + 2*nc]
            page_output = [pred_idx_1, pred_idx_2, pred_idx_3]
            page_id = page_ids[j]
            output_dict[page_id] = page_output
            batch_output.append(page_output)

            curr_idx += num_leafs
            bbox_idx += num_bbox
            
        for batch_id, (val, page_tag_paths) in enumerate(zip(batch_output, tag_paths_segs)):
            page_id = page_ids[batch_id]
            site = pkl2domain[page_id]
            if site not in site_freq_xpath_dict:
                site_freq_xpath_dict[site] = dict()
                site_freq_xpath_dict[site][1] = []
                site_freq_xpath_dict[site][2] = []
                site_freq_xpath_dict[site][3] = []
            for field_idx, node_idx in enumerate(val):
                if field_idx == 0:
                    found_xpath = 0
                    xpath_1 = page_tag_paths[node_idx]
                    for xpath_vote in site_freq_xpath_dict[site][1]:
                        if xpath_vote[0] == xpath_1:
                            found_xpath = 1
                            xpath_vote[1] += 1
                    if found_xpath == 0:
                        site_freq_xpath_dict[site][1].append([xpath_1, 1])
                elif field_idx == 1:
                    found_xpath = 0
                    xpath_2 = page_tag_paths[node_idx]
                    for xpath_vote in site_freq_xpath_dict[site][2]:
                        if xpath_vote[0] == xpath_2:
                            found_xpath = 1
                            xpath_vote[1] += 1
                    if found_xpath == 0:
                        site_freq_xpath_dict[site][2].append([xpath_2, 1])
                elif field_idx == 2:
                    found_xpath = 0
                    xpath_3 = page_tag_paths[node_idx]
                    for xpath_vote in site_freq_xpath_dict[site][3]:
                        if xpath_vote[0] == xpath_3:
                            found_xpath = 1
                            xpath_vote[1] += 1
                    if found_xpath == 0:
                        site_freq_xpath_dict[site][3].append([xpath_3, 1])

    print_pti_acc_stats(gcn_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn')
    
    correct_macro = xpath_voting(site_freq_xpath_dict, label_dict, tag_path_dict, norm_factor_dict, output_dict, pkl2domain)
    print_pti_acc_stats(correct_macro, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='xpath voting')

    return gcn_macro_acc/total_domain_num*100


def match_xpath(logits, labels, match_label, xpaths_segment):
    pred_idx = logits[:, match_label].argmax()
    match_label_indices = (labels == match_label).nonzero()
    if pred_idx in xpaths_segment[match_label-1][1]:
        pass
    else:
        max_score = 0
        for xpath_match in xpaths_segment[match_label-1][1]:
            if logits[xpath_match][match_label] > max_score and xpath_match != -1:
                max_score = logits[xpath_match][match_label]
                pred_idx = xpath_match
    return (pred_idx == match_label_indices[0][0]).item()



def xpath_voting(site_freq_xpath_dict, label_dict, tag_path_dict, norm_factors, significant_value_dict, pkl2domain):
    site_freq_xpaths = dict()
    for site, paths_cnt in site_freq_xpath_dict.items():
        site_freq_xpaths[site] = []
        for field, paths in paths_cnt.items():
            most_freq_path = []
            frequency = 0
            for path in paths:
                count = path[1]
                xpath = path[0]
                if count > frequency:
                    frequency = count
                    most_freq_path = xpath
            site_freq_xpaths[site].append(most_freq_path)

    xpath_vote_output = dict()
    page_indices = list(label_dict.keys())
    for page_idx in page_indices:
        xpath_vote_output[page_idx] = []
        site = pkl2domain[page_idx]
        possible_freq_xpath = site_freq_xpaths[site]
        xpaths = tag_path_dict[page_idx]
        for j, freq_xpath in enumerate(possible_freq_xpath):
            possible_nodes = []
            found = 0
            for i, xpath in enumerate(xpaths):
                if xpath == freq_xpath:
                    possible_nodes.append(i)
                    found += 1
            if found != 1:
                possible_nodes = [significant_value_dict[page_idx][j]]
            xpath_vote_output[page_idx].append(possible_nodes)

    correct_macro = np.zeros(3)
    for page_idx in page_indices:
        labels = label_dict[page_idx]
        xpath_outputs = xpath_vote_output[page_idx]
        norm = norm_factors[page_idx]
        correct_macro[0] += norm*(labels[0] == xpath_outputs[0][0])
        correct_macro[1] += norm*(labels[1] == xpath_outputs[1][0])
        correct_macro[2] += norm*(labels[2] == xpath_outputs[2][0])

    return correct_macro

