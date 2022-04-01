import numpy as np
from time import time
import torch
from models import VAMWODGCN, RCNNGCN
from datasets import GraphConstuctor
from utils import print_and_log
from tqdm import tqdm
import torch.nn.functional as F
import gc

from apex import amp


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

def train_simple_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0


    for epoch in range(1, n_epochs+1):
        model.train()
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += bboxes.shape[0]
            
            optimizer.zero_grad()
            output = model(bboxes.to(device), texts, chars, tag_paths, trees, rel_bboxes.to(device))
            
            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            assert len(output) == sum(leaf_nums)

            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs]
                labels_segment = labels[curr_idx:curr_idx + num_leafs]
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                
            loss = criterion(output, labels)
            epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
    
            loss.backward()
            optimizer.step()


        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            _, class_acc = evaluate_simple_model(model, eval_loader, device, 1, 'VAL', log_file)
            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break

        scheduler.step()

    return best_eval_acc

@torch.no_grad()
def evaluate_simple_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()

    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (_, bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(eval_loader)):
        # if i == 30:
        #     break
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]
        output = model(bboxes.to(device), texts, chars, tag_paths, trees, rel_bboxes.to(device))
        num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
            curr_img_acc = [0]
        
            output_segment = output[curr_idx:curr_idx + num_leafs]
            labels_segment = labels[curr_idx:curr_idx + num_leafs]
            (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                            match(output_segment, labels_segment, 2),
                            match(output_segment, labels_segment, 3)) 
            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic
            img_acc.append(curr_img_acc)

            curr_idx += num_leafs

    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')

    return img_acc, img_macro_acc/total_domain_num * 100


def train_simple_vistext_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0


    for epoch in range(1, n_epochs+1):
        model.train()
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, images, leaf_bboxes, extended_bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            if model.splitted:
                labels = labels.to('cuda:1')
            else:
                labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += leaf_bboxes.shape[0]
            
            optimizer.zero_grad()
            output = model(images.to(device), 
                    leaf_bboxes.to(device), 
                    extended_bboxes.to(device), 
                    texts, chars, tag_paths, trees, 
                    rel_bboxes.to(device))
            num_bboxes = model.num_bboxes_per_img(leaf_bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            assert len(output) == sum(leaf_nums)

            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs]
                labels_segment = labels[curr_idx:curr_idx + num_leafs]
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                
            loss = criterion(output, labels)
            epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
            
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()


        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            _, class_acc = evaluate_simple_vistext_model(model, eval_loader, device, 1, 'VAL', log_file)
            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break

        scheduler.step()

    return best_eval_acc

@torch.no_grad()
def evaluate_simple_vistext_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()

    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (_, images, leaf_bboxes, extended_bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(eval_loader)):
        # if i == 30:
        #     break
        if model.splitted:
            labels = labels.to('cuda:1')
        else:
            labels = labels.to(device) # [total_n_bboxes_in_batch]
        output = model(images.to(device), 
                    leaf_bboxes.to(device), 
                    extended_bboxes.to(device), 
                    texts, chars, tag_paths, trees, 
                    rel_bboxes.to(device))
        num_bboxes = model.num_bboxes_per_img(leaf_bboxes[:, 0])
        curr_idx = 0
        for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
            curr_img_acc = [0]
        
            output_segment = output[curr_idx:curr_idx + num_leafs]
            labels_segment = labels[curr_idx:curr_idx + num_leafs]
            (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                            match(output_segment, labels_segment, 2),
                            match(output_segment, labels_segment, 3)) 
            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic
            img_acc.append(curr_img_acc)

            curr_idx += num_leafs

    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')

    return img_acc, img_macro_acc/total_domain_num * 100


def train_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    TRAIN_GCN = False
    if isinstance(model, VAMWODGCN) or isinstance(model, RCNNGCN):
        TRAIN_GCN = True
        graph_constructor = GraphConstuctor(model.num_candidates, device=device, model=model)
        best_gcn_eval_acc = 0.0
        best_gcn_price_acc = 0.0
        save_file_delimited = model_save_file.split('.')
        gcn_model_save_file = (
                save_file_delimited[0] + '-gcn' + save_file_delimited[1] 
                if len(save_file_delimited) > 1 else 
                save_file_delimited[0] + '-gcn' 
                )

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0
    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0
        if TRAIN_GCN:
            gcn_price_correct, gcn_title_correct, gcn_image_correct = 0.0, 0.0, 0.0
            gcn2_price_correct, gcn2_title_correct, gcn2_image_correct = 0.0, 0.0, 0.0
            gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro = 0.0, 0.0, 0.0
            gcn2_price_correct_macro, gcn2_title_correct_macro, gcn2_image_correct_macro = 0.0, 0.0, 0.0
        for _, images, bboxes, additional_features, context_indices, labels, norm_factors in tqdm(train_loader):
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()

            if TRAIN_GCN:
                encoded_features = model.encode(images.to(device), bboxes.to(device), additional_features.to(device), context_indices.to(device))
                semi_decoded_features = model.semi_decode(encoded_features)
                output = model.final_decode(semi_decoded_features)
            else:
                output = model(images.to(device), bboxes.to(device), additional_features.to(device), context_indices.to(device))

            predictions = output.argmax(dim=1) # [total_n_bboxes_in_batch]

            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)
            for num_bbox, norm_factor in zip(num_bboxes, norm_factors):
                output_segment = output[curr_idx:curr_idx + num_bbox]
                labels_segment = labels[curr_idx:curr_idx + num_bbox]
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor
                curr_idx += num_bbox


            epoch_correct += (predictions == labels).sum().item()
            
            loss = criterion(output, labels)

            assert labels.shape[0] == bboxes.shape[0] and labels.shape[0] == output.shape[0]
            if TRAIN_GCN:
                graph_losses = []
                curr_idx = 0
                for num_bbox, norm_factor in zip(num_bboxes, norm_factors):
                    graph_bboxes = bboxes[curr_idx:curr_idx + num_bbox][:, 1:]
                    # print((predictions[curr_idx:curr_idx + num_bbox] == 3).nonzero())
                    node_features = encoded_features[curr_idx:curr_idx + num_bbox]
                    raw_predicted_prob = output[curr_idx:curr_idx + num_bbox].detach()
                    graph_labels = labels[curr_idx:curr_idx + num_bbox]
                    graph_and_tmp_labels_and_select_indices = graph_constructor.construct(
                        graph_bboxes, raw_predicted_prob, graph_labels, train=True, node_features=node_features, 
                        return_select_indices=True)
                    if graph_and_tmp_labels_and_select_indices is not None:
                        graph, tmp_labels, select_indices = graph_and_tmp_labels_and_select_indices
                        logits, logits2 = model.gcn_forward(graph, node_features[select_indices])
                        graph_loss = F.cross_entropy(logits, tmp_labels)
                        combined_losses = F.cross_entropy(logits2, tmp_labels)
                        graph_losses.append((graph_loss + combined_losses) * 0.5) # * norm_factor)
                        # graph_losses.append(graph_loss)

                        # print(logits[:model.num_candidates], tmp_labels[:model.num_candidates])
                        nc = model.num_candidates
                        (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                                match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                                match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))
                        (pc2, tc2, ic2) = (match(logits2[:nc], tmp_labels[:nc], 1),
                                match(logits2[nc:2*nc], tmp_labels[nc:2*nc], 2),
                                match(logits2[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))


                        gcn_price_correct += pc
                        gcn_price_correct_macro += pc * norm_factor
                        gcn_title_correct += tc
                        gcn_title_correct_macro += tc * norm_factor
                        gcn_image_correct += ic
                        gcn_image_correct_macro += ic * norm_factor

                        gcn2_price_correct += pc2
                        gcn2_price_correct_macro += pc2 * norm_factor
                        gcn2_title_correct += tc2
                        gcn2_title_correct_macro += tc2 * norm_factor
                        gcn2_image_correct += ic2
                        gcn2_image_correct_macro += ic2 * norm_factor
                    curr_idx += num_bbox
                if len(graph_losses) > 0:
                    #TODO: Try different scale
                    total_graph_loss = sum(graph_losses) * (1/len(graph_losses))  #* len(eval_loader.dataset) / total_domain_num
                    loss += total_graph_loss

            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  Accuracy: %.2f%%  (%.2fs)' % (epoch, epoch_loss/n_bboxes, 100*epoch_correct/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        if TRAIN_GCN:
            print_pti_acc_stats([gcn_price_correct, gcn_title_correct, gcn_image_correct], n_webpages, log_file, acc_modifier='gcn')
            print_pti_acc_stats([gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn')
            print_pti_acc_stats([gcn2_price_correct, gcn2_title_correct, gcn2_image_correct], n_webpages, log_file, acc_modifier='gcn2')
            print_pti_acc_stats([gcn2_price_correct_macro, gcn2_title_correct_macro, gcn2_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn2')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            if TRAIN_GCN:
                _, class_acc, _, gcn_class_acc = evaluate_model(model, eval_loader, device, 1, 'VAL', log_file)
                gcn_eval_acc = gcn_class_acc.mean()
                if gcn_eval_acc > best_gcn_eval_acc:
                    print('GCN Model Saved!', gcn_eval_acc, '>', best_gcn_eval_acc)
                    best_gcn_eval_acc = gcn_eval_acc
                    torch.save(model.state_dict(), gcn_model_save_file)
            else:
                _, class_acc = evaluate_model(model, eval_loader, device, 1, 'VAL', log_file)
            eval_acc = class_acc.mean() # only consider non-BG classes
            model.train()

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
            model.train()

        scheduler.step()

    if TRAIN_GCN:
        return best_eval_acc, gcn_model_save_file

    return best_eval_acc

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
            # print(probs[child.idx], max_likelihood)
            # print((probs[child.idx][target_label_pos] > max_likelihood).item())
            if (probs[child.idx][target_label_pos] > max_likelihood).item():
                max_likelihood = probs[child.idx][target_label_pos].item()
                max_idx = i
        return tree_match(tree.children[max_idx], probs, true_labels, target_label_pos)
    # print(tree.idx)
    # print(true_labels)
    # print(target_label_pos)
    # print(true_labels == target_label_pos)
    # print((true_labels == target_label_pos).nonzero())
    # print((tree.idx == (true_labels == target_label_pos).nonzero()[0][0]).item())
    return (tree.idx == (true_labels == target_label_pos).nonzero()[0][0]).item()

def convert_labels(labels, leaf_num):
    return torch.argmax(labels[:leaf_num], dim=1)


def train_tree_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0
    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0
        for i, (_, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors) in enumerate(tqdm(train_loader)):
            # if i == 30:
            #     break
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()

            output = model(images.to(device), bboxes.to(device), additional_features.to(device), trees)

            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)
            # squashed_labelss=[]
            # squashed_probss=[]
            # labels_for_loss = []
            for i, (num_bbox, norm_factor) in enumerate(zip(num_bboxes, norm_factors)):
                output_segment = output[curr_idx:curr_idx + num_bbox]
                labels_segment = labels[curr_idx:curr_idx + num_bbox]

                # labels_for_loss.append(labels_segment[:leaf_nums[i]])
                squashed_labels = convert_labels(labels_segment, leaf_nums[i])
                # squashed_probs = tree_squash_probs(trees[i], output_segment, leaf_nums[i], device)
                # (pc, tc, ic) = (match(squashed_probs, squashed_labels, 1),
                #                 match(squashed_probs, squashed_labels, 2),
                #                 match(squashed_probs, squashed_labels, 3))
                (pc, tc, ic) = (tree_match(trees[i], output_segment, squashed_labels, 1),
                                tree_match(trees[i], output_segment, squashed_labels, 2),
                                tree_match(trees[i], output_segment, squashed_labels, 3))              
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor
                curr_idx += num_bbox
                # squashed_labelss.append(squashed_labels)
                # squashed_probss.append(squashed_probs)

            # final_labels = torch.cat(squashed_labelss, dim=0)
            # final_outputs = torch.cat(squashed_probss, dim=0)
            # labels_for_loss = torch.cat(labels_for_loss, dim=0).float()
            # predictions = final_outputs.argmax(dim=1) # [total_n_bboxes_in_batch]

            # epoch_correct += (predictions == final_labels).sum().item()
            
            # loss = criterion(final_outputs.view(-1), labels_for_loss.view(-1))
            # print()
            loss = criterion(output.view(-1), labels.view(-1))
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            _, class_acc = evaluate_tree_model(model, eval_loader, device, 1, 'VAL', log_file)
            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
            model.train()

        scheduler.step()

    return best_eval_acc

def train_pass_down_tree_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0
    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0
        for i, (_, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors) in enumerate(tqdm(train_loader)):
            # if i == 30:
            #     break
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()

            output = model(images.to(device), bboxes.to(device), additional_features.to(device), trees)

            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            curr_label_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            # print(len(output), sum(leaf_nums), leaf_nums)
            assert len(output) == sum(leaf_nums)

            labels_for_loss = []
            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs]
                labels_segment = labels[curr_label_idx:curr_label_idx + num_leafs]
                converted_label_segment = convert_labels(labels_segment, num_leafs)
                (pc, tc, ic) = (match(output_segment, converted_label_segment, 1),
                                match(output_segment, converted_label_segment, 2),
                                match(output_segment, converted_label_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor
                curr_idx += num_leafs
                curr_label_idx += num_bbox
                labels_for_loss.append(converted_label_segment)

            # final_labels = torch.cat(squashed_labelss, dim=0)
            # final_outputs = torch.cat(squashed_probss, dim=0)
            labels_for_loss = torch.cat(labels_for_loss)
            # predictions = final_outputs.argmax(dim=1) # [total_n_bboxes_in_batch]

            # epoch_correct += (predictions == final_labels).sum().item()
            
            # loss = criterion(final_outputs.view(-1), labels_for_loss.view(-1))
            # print()
            loss = criterion(output, labels_for_loss)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            _, class_acc = evaluate_pass_down_tree_model(model, eval_loader, device, 1, 'VAL', log_file)
            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
            model.train()

        scheduler.step()

    return best_eval_acc

def train_final_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
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
    # print(model_save_file, save_file_delimited, gcn_model_save_file)

    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0

        tree_price_correct, tree_title_correct, tree_image_correct = 0.0, 0.0, 0.0
        tree_price_correct_macro, tree_title_correct_macro, tree_image_correct_macro = 0.0, 0.0, 0.0

        gcn_price_correct, gcn_title_correct, gcn_image_correct = 0.0, 0.0, 0.0
        gcn2_price_correct, gcn2_title_correct, gcn2_image_correct = 0.0, 0.0, 0.0
        gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro = 0.0, 0.0, 0.0
        gcn2_price_correct_macro, gcn2_title_correct_macro, gcn2_image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            # if i == 30:
            #     break
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()

            encoded_features, tree_encoded_features = model.encode(images.to(device), bboxes.to(device), additional_features.to(device), trees, rel_bboxes.to(device))
            output = model.decode(encoded_features)
            # tree_output = model.tree_decode(tree_encoded_features)
            tree_output = output
            
            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            curr_label_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            # print(len(output), sum(leaf_nums), leaf_nums)
            assert len(output) == sum(leaf_nums)

            labels_for_loss = []
            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs]
                tree_output_segment = tree_output[curr_idx:curr_idx + num_leafs]
                labels_segment = labels[curr_label_idx:curr_label_idx + num_leafs]
                converted_label_segment = convert_labels(labels_segment, num_leafs)
                (pc, tc, ic) = (match(output_segment, converted_label_segment, 1),
                                match(output_segment, converted_label_segment, 2),
                                match(output_segment, converted_label_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor

                (pc, tc, ic) = (match(tree_output_segment, converted_label_segment, 1),
                                match(tree_output_segment, converted_label_segment, 2),
                                match(tree_output_segment, converted_label_segment, 3))            
                tree_price_correct += pc
                tree_price_correct_macro += pc * norm_factor
                tree_title_correct += tc
                tree_title_correct_macro += tc * norm_factor
                tree_image_correct += ic
                tree_image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                curr_label_idx += num_bbox
                labels_for_loss.append(converted_label_segment)

            labels_for_loss = torch.cat(labels_for_loss)

            loss = criterion(output, labels_for_loss)
            # tree_loss = criterion(tree_output, labels_for_loss)
            # loss += tree_loss

            graph_losses = []
            curr_idx = 0
            bbox_idx = 0
            for num_leafs, num_bbox, norm_factor in zip(leaf_nums, num_bboxes, norm_factors):
                graph_bboxes = bboxes[bbox_idx:bbox_idx + num_leafs][:, 1:]
                # print((predictions[curr_idx:curr_idx + num_bbox] == 3).nonzero())
                node_features = tree_encoded_features[curr_idx:curr_idx + num_leafs].detach()
                raw_predicted_prob = tree_output[curr_idx:curr_idx + num_leafs].detach()
                graph_labels = labels_for_loss[curr_idx:curr_idx + num_leafs]
                graph_and_tmp_labels_and_select_indices = graph_constructor.construct(
                    graph_bboxes, raw_predicted_prob, graph_labels, train=True, node_features=node_features, 
                    return_select_indices=True)
                if graph_and_tmp_labels_and_select_indices is not None:
                    graph, tmp_labels, select_indices = graph_and_tmp_labels_and_select_indices
                    logits, logits2 = model.gcn_forward(graph, node_features[select_indices])
                    graph_loss = F.cross_entropy(logits, tmp_labels)
                    combined_losses = F.cross_entropy(logits2, tmp_labels)
                    graph_losses.append((graph_loss + combined_losses)) # * norm_factor)
                    # graph_losses.append(graph_loss)

                    # print(logits[:model.num_candidates], tmp_labels[:model.num_candidates])
                    nc = model.num_candidates
                    (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))
                    (pc2, tc2, ic2) = (match(logits2[:nc], tmp_labels[:nc], 1),
                            match(logits2[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits2[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))


                    gcn_price_correct += pc
                    gcn_price_correct_macro += pc * norm_factor
                    gcn_title_correct += tc
                    gcn_title_correct_macro += tc * norm_factor
                    gcn_image_correct += ic
                    gcn_image_correct_macro += ic * norm_factor

                    gcn2_price_correct += pc2
                    gcn2_price_correct_macro += pc2 * norm_factor
                    gcn2_title_correct += tc2
                    gcn2_title_correct_macro += tc2 * norm_factor
                    gcn2_image_correct += ic2
                    gcn2_image_correct_macro += ic2 * norm_factor
                curr_idx += num_leafs
                bbox_idx += num_bbox

            if len(graph_losses) > 0:
                #TODO: Try different scale
                total_graph_loss = sum(graph_losses) * (1/len(graph_losses))  #* len(eval_loader.dataset) / total_domain_num
                loss += total_graph_loss


            epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
            
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        print_pti_acc_stats([tree_price_correct, tree_title_correct, tree_image_correct], n_webpages, log_file, acc_modifier='tree')
        print_pti_acc_stats([tree_price_correct_macro, tree_title_correct_macro, tree_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='tree')
        print_pti_acc_stats([gcn_price_correct, gcn_title_correct, gcn_image_correct], n_webpages, log_file, acc_modifier='gcn')
        print_pti_acc_stats([gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn')
        print_pti_acc_stats([gcn2_price_correct, gcn2_title_correct, gcn2_image_correct], n_webpages, log_file, acc_modifier='gcn2')
        print_pti_acc_stats([gcn2_price_correct_macro, gcn2_title_correct_macro, gcn2_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn2')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            _, class_acc, _, gcn_class_acc = evaluate_final_model(model, eval_loader, device, 1, 'VAL', log_file)
            gcn_eval_acc = gcn_class_acc.mean()
            if gcn_eval_acc > best_gcn_eval_acc:
                print('GCN Model Saved!', gcn_eval_acc, '>', best_gcn_eval_acc)
                best_gcn_eval_acc = gcn_eval_acc
                torch.save(model.state_dict(), gcn_model_save_file)

            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
            model.train()

        scheduler.step()

    return best_eval_acc, gcn_model_save_file





def train_dummy_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    total_domain_num = train_loader.dataset.total_domain_num
    n_webpages = 0

    best_gcn_eval_acc = 0.0
    save_file_delimited = model_save_file.split('/')
    prev_path = "/".join(save_file_delimited[:-1])
    save_file_delimited = save_file_delimited[-1].split('.')
    gcn_model_save_file = (
            prev_path + "/" + save_file_delimited[0] + '-gcn.' + save_file_delimited[1]
            if len(save_file_delimited) > 1 else 
            prev_path + "/" + save_file_delimited[0] + '-gcn' 
            )
    # print(model_save_file, save_file_delimited, gcn_model_save_file)

    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            # if i == 30:
            #     break
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()
            encoded_features = model.encode(bboxes.to(device))
            output = model.decode(encoded_features)
            tree_output = output
            
            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            # print(len(output), sum(leaf_nums), leaf_nums)
            assert len(output) == sum(leaf_nums)

            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs]
                labels_segment = labels[curr_idx:curr_idx + num_leafs]
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor

                curr_idx += num_leafs


            loss = criterion(output, labels)
            # tree_loss = criterion(tree_output, labels_for_loss)
            # loss += tree_loss


            epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
            
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        # if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
        #     _, class_acc, _, gcn_class_acc = evaluate_final_model(model, eval_loader, device, 1, 'VAL', log_file)
        #     gcn_eval_acc = gcn_class_acc.mean()
        #     if gcn_eval_acc > best_gcn_eval_acc:
        #         print('GCN Model Saved!', gcn_eval_acc, '>', best_gcn_eval_acc)
        #         best_gcn_eval_acc = gcn_eval_acc
        #         torch.save(model.state_dict(), gcn_model_save_file)

        #     eval_acc = class_acc.mean() # only consider non-BG classes

        #     if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
        #         print('Model Saved!', eval_acc, '>', best_eval_acc)
        #         best_eval_acc = eval_acc
        #         patience_count = 0
        #         torch.save(model.state_dict(), model_save_file)
        #     else:
        #         patience_count += 1
        #         if patience_count >= patience:
        #             print('Early Stopping!')
        #             break
        #     model.train()

        scheduler.step()

    return best_eval_acc, gcn_model_save_file





def train_text_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
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
    # print(model_save_file, save_file_delimited, gcn_model_save_file)

    for epoch in range(1, n_epochs+1):
        torch.cuda.empty_cache()
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0

        pre_rnn_price_correct, pre_rnn_title_correct, pre_rnn_image_correct = 0.0, 0.0, 0.0
        pre_rnn_price_correct_macro, pre_rnn_title_correct_macro, pre_rnn_image_correct_macro = 0.0, 0.0, 0.0

        gcn_price_correct, gcn_title_correct, gcn_image_correct = 0.0, 0.0, 0.0
        gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            # if i == 30:
            #     break
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += bboxes.shape[0]
            
            optimizer.zero_grad()


            # with profiler.profile(record_shapes=True) as prof:
            #     with profiler.record_function("model_inference"):
            #         encoded_features, tree_encoded_features = model.encode(bboxes.to(device), texts, chars, tag_paths, trees, rel_bboxes.to(device))
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            encoded_features, pre_leaf_rnn_output = model.encode(bboxes.to(device), texts, chars, tag_paths, trees, rel_bboxes.to(device))
            output = model.decode(encoded_features)
            tree_output = output
            
            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            if epoch == 1:
                n_webpages += len(num_bboxes)

            # print(len(output), sum(leaf_nums), leaf_nums)
            assert len(output) == sum(leaf_nums)

            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs]
                labels_segment = labels[curr_idx:curr_idx + num_leafs]
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor

                pre_rnn_output_segment = pre_leaf_rnn_output[curr_idx:curr_idx + num_leafs]
                (pc, tc, ic) = (match(pre_rnn_output_segment, labels_segment, 1),
                                match(pre_rnn_output_segment, labels_segment, 2),
                                match(pre_rnn_output_segment, labels_segment, 3))            
                pre_rnn_price_correct += pc
                pre_rnn_price_correct_macro += pc * norm_factor
                pre_rnn_title_correct += tc
                pre_rnn_title_correct_macro += tc * norm_factor
                pre_rnn_image_correct += ic
                pre_rnn_image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                
            loss = criterion(output, labels) + criterion(pre_leaf_rnn_output, labels)
            # tree_loss = criterion(tree_output, labels_for_loss)
            # loss += tree_loss

            graph_losses = []
            curr_idx = 0
            bbox_idx = 0
            for num_leafs, num_bbox, norm_factor in zip(leaf_nums, num_bboxes, norm_factors):
                graph_bboxes = bboxes[bbox_idx:bbox_idx + num_leafs][:, 1:]
                node_features = encoded_features[curr_idx:curr_idx + num_leafs]
                raw_predicted_prob = output[curr_idx:curr_idx + num_leafs].detach()
                graph_labels = labels[curr_idx:curr_idx + num_leafs]
                graph_and_tmp_labels_and_select_indices = graph_constructor.construct(
                    graph_bboxes, raw_predicted_prob, graph_labels, train=True, node_features=node_features, 
                    return_select_indices=True)
                if graph_and_tmp_labels_and_select_indices is not None:
                    graph, tmp_labels, select_indices = graph_and_tmp_labels_and_select_indices
                    logits = model.gcn_forward(graph, node_features[select_indices])
                    graph_loss = F.cross_entropy(logits, tmp_labels)
                    graph_losses.append(graph_loss) # * norm_factor)
                    # graph_losses.append(graph_loss)

                    # print(logits[:model.num_candidates], tmp_labels[:model.num_candidates])
                    nc = model.num_candidates
                    (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))

                    gcn_price_correct += pc
                    gcn_price_correct_macro += pc * norm_factor
                    gcn_title_correct += tc
                    gcn_title_correct_macro += tc * norm_factor
                    gcn_image_correct += ic
                    gcn_image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                bbox_idx += num_bbox

            if len(graph_losses) > 0:
                #TODO: Try different scale
                total_graph_loss = sum(graph_losses) * (1/len(graph_losses))  #* len(eval_loader.dataset) / total_domain_num
                loss += total_graph_loss

            # if len(norm_factors) == 1:
            #     loss *= norm_factors[0] * len(train_loader.dataset)  / total_domain_num 
            epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
    
            loss.backward()
            optimizer.step()
        
        # To maximize memory save
        del bboxes
        del texts
        del chars
        del tag_paths
        del labels
        del trees
        del leaf_nums
        del norm_factors
        del rel_bboxes
        del encoded_features
        del output
        del pre_leaf_rnn_output
        del total_graph_loss
        del loss

        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        print_pti_acc_stats([pre_rnn_price_correct, pre_rnn_title_correct, pre_rnn_image_correct], n_webpages, log_file, acc_modifier='pre_rnn')
        print_pti_acc_stats([pre_rnn_price_correct_macro, pre_rnn_title_correct_macro, pre_rnn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='pre_rnn')
        print_pti_acc_stats([gcn_price_correct, gcn_title_correct, gcn_image_correct], n_webpages, log_file, acc_modifier='gcn')
        print_pti_acc_stats([gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            torch.cuda.empty_cache()
            _, class_acc, _, gcn_class_acc = evaluate_text_model(model, eval_loader, device, 1, 'VAL', log_file)
            gcn_eval_acc = gcn_class_acc.mean()
            if gcn_eval_acc > best_gcn_eval_acc:
                print('GCN Model Saved!', gcn_eval_acc, '>', best_gcn_eval_acc)
                best_gcn_eval_acc = gcn_eval_acc
                torch.save(model.state_dict(), gcn_model_save_file)

            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
            model.train()

        scheduler.step()

    return best_eval_acc, gcn_model_save_file

def train_vistext_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
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
    # print(model_save_file, save_file_delimited, gcn_model_save_file)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, n_epochs+1):
#         torch.cuda.empty_cache()
#         gc.collect()
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        price_correct, title_correct, image_correct = 0.0, 0.0, 0.0
        price_correct_macro, title_correct_macro, image_correct_macro = 0.0, 0.0, 0.0
        gcn_price_correct, gcn_title_correct, gcn_image_correct = 0.0, 0.0, 0.0
        gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro = 0.0, 0.0, 0.0

        for i, (_, images, leaf_bboxes, extended_bboxes, texts, chars, leaf_tags, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(train_loader)):
            
            if model.splitted:
                labels = labels.to('cuda:1')
            else:
                labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += leaf_bboxes.shape[0]
            
            optimizer.zero_grad()

            # try:
            # with torch.cuda.amp.autocast():
            #     encoded_features = model.encode(
            #         images.to(device).half(), 
            #         leaf_bboxes.to(device).half(), 
            #         extended_bboxes.to(device).half(), 
            #         texts, chars, tag_paths, trees, 
            #         rel_bboxes.to(device).half())
            #     output = model.decode(encoded_features)
            
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

            # print(len(output), sum(leaf_nums), leaf_nums)
            assert len(output) == sum(leaf_nums)

            for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
                output_segment = output[curr_idx:curr_idx + num_leafs].detach()
                labels_segment = labels[curr_idx:curr_idx + num_leafs].detach()
                (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                                match(output_segment, labels_segment, 2),
                                match(output_segment, labels_segment, 3))            
                price_correct += pc
                price_correct_macro += pc * norm_factor
                title_correct += tc
                title_correct_macro += tc * norm_factor
                image_correct += ic
                image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
            
#             with torch.cuda.amp.autocast():
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
                    graph_loss = F.cross_entropy(logits, tmp_labels)
                    graph_losses.append(graph_loss) # * norm_factor)
                    # graph_losses.append(graph_loss)

                    # print(logits[:model.num_candidates], tmp_labels[:model.num_candidates])
                    nc = model.num_candidates
                    (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))


                    gcn_price_correct += pc
                    gcn_price_correct_macro += pc * norm_factor
                    gcn_title_correct += tc
                    gcn_title_correct_macro += tc * norm_factor
                    gcn_image_correct += ic
                    gcn_image_correct_macro += ic * norm_factor

                curr_idx += num_leafs
                bbox_idx += num_bbox

            if len(graph_losses) > 0:
                #TODO: Try different scale
                total_graph_loss = sum(graph_losses) * (1/len(graph_losses))  #* len(eval_loader.dataset) / total_domain_num
                loss += total_graph_loss
                # total_graph_loss.backward()
                # del total_graph_loss

            epoch_loss += loss.item()  # There is normal loss + tree loss + and 2 gcn loss
            
#             del images
#             del leaf_bboxes
#             del extended_bboxes
#             del texts
#             del chars
#             del tag_paths
#             del labels
#             del trees
#             del leaf_nums
#             del norm_factors
#             del rel_bboxes
#             del encoded_features
#             del output
            # del total_graph_loss

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            # except Exception as e:
            #     print(e)
            #     print('Error encountered, skip batch')
            #     continue
        # To maximize memory save
#         del loss
        print_and_log('Epoch: %2d  Loss: %.4f  (%.2fs)' % (epoch, epoch_loss/n_bboxes, time()-start), log_file)
#         del epoch_loss
#         gc.collect()
        print_pti_acc_stats([price_correct, title_correct, image_correct], n_webpages, log_file)
        print_pti_acc_stats([price_correct_macro, title_correct_macro, image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc')
        # print_pti_acc_stats([pre_rnn_price_correct, pre_rnn_title_correct, pre_rnn_image_correct], n_webpages, log_file, acc_modifier='pre_rnn')
        # print_pti_acc_stats([pre_rnn_price_correct_macro, pre_rnn_title_correct_macro, pre_rnn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='pre_rnn')
        print_pti_acc_stats([gcn_price_correct, gcn_title_correct, gcn_image_correct], n_webpages, log_file, acc_modifier='gcn')
        print_pti_acc_stats([gcn_price_correct_macro, gcn_title_correct_macro, gcn_image_correct_macro], total_domain_num, log_file, acc_name='Macro Acc', acc_modifier='gcn')
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            torch.cuda.empty_cache()
            _, class_acc, _, gcn_class_acc = evaluate_vistext_model(model, eval_loader, device, 1, 'VAL', log_file)
            gcn_eval_acc = gcn_class_acc.mean()
            if gcn_eval_acc > best_gcn_eval_acc:
                print('GCN Model Saved!', gcn_eval_acc, '>', best_gcn_eval_acc)
                best_gcn_eval_acc = gcn_eval_acc
                torch.save(model.state_dict(), gcn_model_save_file)

            eval_acc = class_acc.mean() # only consider non-BG classes

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                print('Model Saved!', eval_acc, '>', best_eval_acc)
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
            model.train()
#             del _
#             del class_acc
#             del gcn_class_acc

        scheduler.step()

    return best_eval_acc, gcn_model_save_file


@torch.no_grad()
def evaluate_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()

    EVAL_GCN = False
    graph_constructor = None
    if isinstance(model, VAMWODGCN) or isinstance(model, RCNNGCN):
        EVAL_GCN = True
        graph_constructor = GraphConstuctor(model.num_candidates, device=device, model=model)


    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    gcn_acc = []
    gcn_macro_acc = np.array([0.0, 0.0, 0.0])
    gcn2_acc = []
    gcn2_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (img_ids, images, bboxes, additional_features, context_indices, labels, norm_factors) in enumerate(tqdm(eval_loader)):

        labels = labels.to(device) # [total_n_bboxes_in_batch]
        if EVAL_GCN:
            encoded_features = model.encode(images.to(device), bboxes.to(device), additional_features.to(device), context_indices.to(device))
            semi_decoded_features = model.semi_decode(encoded_features)
            output = model.final_decode(semi_decoded_features)
        else:
            output = model(images.to(device), bboxes.to(device), additional_features.to(device), context_indices.to(device)) # [total_n_bboxes_in_batch, n_classes]
        
        if EVAL_GCN:
            num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
            curr_idx = 0
            for num_bbox, norm_factor in zip(num_bboxes, norm_factors):
                curr_gcn_acc = []
                curr_gcn2_acc = []

                graph_bboxes = bboxes[curr_idx:curr_idx + num_bbox][:, 1:]
                graph_labels = labels[curr_idx:curr_idx + num_bbox]
                raw_predicted_prob = output[curr_idx:curr_idx + num_bbox]
                node_features = encoded_features[curr_idx:curr_idx + num_bbox]
                graph, tmp_labels, select_indices = graph_constructor.construct(
                    graph_bboxes, raw_predicted_prob, graph_labels, train=False, node_features=node_features, 
                    return_select_indices=True)
                logits, logits2 = model.gcn_forward(graph, node_features[select_indices])

                nc = model.num_candidates

                (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                                match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                                match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))
                (pc2, tc2, ic2) = (match(logits2[:nc], tmp_labels[:nc], 1),
                                match(logits2[nc:2*nc], tmp_labels[nc:2*nc], 2),
                                match(logits2[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))

                curr_gcn_acc.append(pc)
                curr_gcn2_acc.append(pc2)

                curr_gcn_acc.append(tc)
                curr_gcn2_acc.append(tc2)

                curr_gcn_acc.append(ic)
                curr_gcn2_acc.append(ic2)
                
                gcn_macro_acc[0] += norm_factor * pc
                gcn_macro_acc[1] += norm_factor * tc
                gcn_macro_acc[2] += norm_factor * ic

                gcn2_macro_acc[0] += norm_factor * pc2
                gcn2_macro_acc[1] += norm_factor * tc2
                gcn2_macro_acc[2] += norm_factor * ic2

                gcn_acc.append(curr_gcn_acc)
                gcn2_acc.append(curr_gcn2_acc)
                curr_idx += num_bbox

        batch_indices = torch.unique(bboxes[:,0]).long()
        for index in batch_indices: # for each image
            img_id = img_ids[index]
            norm_factor = norm_factors[index]
            img_indices = (bboxes[:,0] == index)
            labels_img = labels[img_indices].view(-1,1)
            output_img = output[img_indices]

            label_indices = torch.arange(labels_img.shape[0], device=device).view(-1,1)
            indexed_labels = torch.cat((label_indices, labels_img), dim=1)
            indexed_labels = indexed_labels[indexed_labels[:,-1] != 0] # labels for bbox other than BG
            
            curr_img_acc = [img_id] # [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
            if use_softmax:
                output_img = torch.softmax(output_img, dim=1)
            if k > 0:
                top_k_predictions = torch.argsort(output_img, dim=0)[output_img.shape[0]-k:] # [k, n_classes] indices indicating top k predicted bbox
                # print(top_k_predictions)
                for c in range(1, n_classes):
#                     print(indexed_labels[indexed_labels[:,-1] == c])
#                     print(img_id)
                    true_bbox = indexed_labels[indexed_labels[:,-1] == c][0,0]
                    pred_bboxes = top_k_predictions[:, c]
                    curr_img_acc.append(1 if true_bbox in pred_bboxes else 0)
                    img_macro_acc[c - 1] += norm_factor if true_bbox in pred_bboxes else 0
            else:
                predictions = torch.argmax(output_img, dim=1)
                for c in range(1, n_classes):
                    true_bbox = indexed_labels[indexed_labels[:,-1] == c][0,0]
                    pred_bboxes = (predictions == c).nonzero()[:,0]
                    # print(true_bbox, pred_bboxes)
                    curr_img_acc.append(1 if true_bbox in pred_bboxes else 0)
                    img_macro_acc[c - 1] += norm_factor if true_bbox in pred_bboxes else 0
            img_acc.append(curr_img_acc)
        
    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')

    if EVAL_GCN:
        gcn_acc = np.array(gcn_acc, dtype=np.int32)
        gcn_class_acc = gcn_acc.mean(0) * 100
        print_and_log('GCN Avg_class_Accuracy: %.2f%%' % (gcn_class_acc.mean()), log_file)
        for c in range(1, n_classes):
            print_and_log('%s gcn-Acc: %.2f%%' % (model.class_names[c], gcn_class_acc[c-1]), log_file)
        print_and_log('', log_file)

        print_pti_acc_stats(gcn_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn')

        gcn2_acc = np.array(gcn2_acc, dtype=np.int32)
        gcn2_class_acc = gcn2_acc.mean(0) * 100
        print_and_log('GCN + GRU Hidden Combined Avg_class_Accuracy: %.2f%%' % (gcn2_class_acc.mean()), log_file)
        for c in range(1, n_classes):
            print_and_log('%s gcn2-Acc: %.2f%%' % (model.class_names[c], gcn2_class_acc[c-1]), log_file)
        print_and_log('', log_file)

        print_pti_acc_stats(gcn2_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn2')

        return img_acc, img_macro_acc/total_domain_num * 100, gcn2_acc, gcn2_macro_acc/total_domain_num*100

    return img_acc, img_macro_acc/total_domain_num * 100

@torch.no_grad()
def evaluate_tree_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()

    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (img_ids, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors) in enumerate(tqdm(eval_loader)):
        # if i == 30:
        #     break
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]
        output = model(images.to(device), bboxes.to(device), additional_features.to(device), trees)

        num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        for i, (num_bbox, norm_factor) in enumerate(zip(num_bboxes, norm_factors)):
            curr_img_acc = [0]

            output_segment = output[curr_idx:curr_idx + num_bbox]
            labels_segment = labels[curr_idx:curr_idx + num_bbox]

            squashed_labels = convert_labels(labels_segment, leaf_nums[i])
            # squashed_probs = tree_squash_probs(trees[i], output_segment, leaf_nums[i], device)
            # (pc, tc, ic) = (match(squashed_probs, squashed_labels, 1),
            #                 match(squashed_probs, squashed_labels, 2),
            #                 match(squashed_probs, squashed_labels, 3))

            (pc, tc, ic) = (tree_match(trees[i], output_segment, squashed_labels, 1),
                            tree_match(trees[i], output_segment, squashed_labels, 2),
                            tree_match(trees[i], output_segment, squashed_labels, 3)) 

            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic

            img_acc.append(curr_img_acc)
            curr_idx += num_bbox
        
    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')

    return img_acc, img_macro_acc/total_domain_num * 100


@torch.no_grad()
def evaluate_pass_down_tree_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()

    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (img_ids, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors) in enumerate(tqdm(eval_loader)):
        # if i == 30:
        #     break
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]
        output = model(images.to(device), bboxes.to(device), additional_features.to(device), trees)

        num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        curr_label_idx = 0
        for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
            curr_img_acc = [0]

            output_segment = output[curr_idx:curr_idx + num_leafs]
            labels_segment = labels[curr_label_idx:curr_label_idx + num_leafs]
            converted_label_segment = convert_labels(labels_segment, num_leafs)
            (pc, tc, ic) = (match(output_segment, converted_label_segment, 1),
                            match(output_segment, converted_label_segment, 2),
                            match(output_segment, converted_label_segment, 3))  

            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic

            img_acc.append(curr_img_acc)
            curr_idx += num_leafs
            curr_label_idx += num_bbox
        
    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')

    return img_acc, img_macro_acc/total_domain_num * 100

@torch.no_grad()
def evaluate_final_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
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

    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    tree_img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    tree_img_macro_acc = np.array([0.0, 0.0, 0.0])
    gcn_acc = []
    gcn_macro_acc = np.array([0.0, 0.0, 0.0])
    gcn2_acc = []
    gcn2_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (img_ids, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(eval_loader)):
        # if i == 30:
        #     break
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]

        encoded_features, tree_encoded_features = model.encode(images.to(device), bboxes.to(device), additional_features.to(device), trees, rel_bboxes.to(device))
        output = model.decode(encoded_features)
        # tree_output = model.tree_decode(tree_encoded_features)
        # output = model(images.to(device), bboxes.to(device), additional_features.to(device), trees)

        num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        curr_label_idx = 0
        labels_for_loss = []
        for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
            curr_img_acc = [0]
            tree_curr_img_acc = [0]

            output_segment = output[curr_idx:curr_idx + num_leafs]
            tree_output_segment = tree_output[curr_idx:curr_idx + num_leafs]
            labels_segment = labels[curr_label_idx:curr_label_idx + num_leafs]
            converted_label_segment = convert_labels(labels_segment, num_leafs)
            (pc, tc, ic) = (match(output_segment, converted_label_segment, 1),
                            match(output_segment, converted_label_segment, 2),
                            match(output_segment, converted_label_segment, 3))  

            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic

            img_acc.append(curr_img_acc)

            (pc, tc, ic) = (match(tree_output_segment, converted_label_segment, 1),
                            match(tree_output_segment, converted_label_segment, 2),
                            match(tree_output_segment, converted_label_segment, 3))  
            tree_curr_img_acc.append(pc)
            tree_curr_img_acc.append(tc)
            tree_curr_img_acc.append(ic)
            
            tree_img_macro_acc[0] += norm_factor * pc
            tree_img_macro_acc[1] += norm_factor * tc
            tree_img_macro_acc[2] += norm_factor * ic

            tree_img_acc.append(tree_curr_img_acc)

            curr_idx += num_leafs
            curr_label_idx += num_bbox
            labels_for_loss.append(converted_label_segment)

        labels_for_loss = torch.cat(labels_for_loss)

        curr_idx = 0
        bbox_idx = 0
        for num_leafs, num_bbox, norm_factor in zip(leaf_nums, num_bboxes, norm_factors):
            curr_gcn_acc = []
            curr_gcn2_acc = []

            graph_bboxes = bboxes[bbox_idx:bbox_idx + num_leafs][:, 1:]
            graph_labels = labels_for_loss[curr_idx:curr_idx + num_leafs]
            raw_predicted_prob = output[curr_idx:curr_idx + num_leafs]
            node_features = encoded_features[curr_idx:curr_idx + num_leafs].detach()
            graph, tmp_labels, select_indices = graph_constructor.construct(
                graph_bboxes, raw_predicted_prob, graph_labels, train=False, node_features=node_features, 
                return_select_indices=True)
            logits, logits2 = model.gcn_forward(graph, node_features[select_indices])

            nc = model.num_candidates

            (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))
            (pc2, tc2, ic2) = (match(logits2[:nc], tmp_labels[:nc], 1),
                            match(logits2[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits2[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))

            curr_gcn_acc.append(pc)
            curr_gcn2_acc.append(pc2)

            curr_gcn_acc.append(tc)
            curr_gcn2_acc.append(tc2)

            curr_gcn_acc.append(ic)
            curr_gcn2_acc.append(ic2)
            
            gcn_macro_acc[0] += norm_factor * pc
            gcn_macro_acc[1] += norm_factor * tc
            gcn_macro_acc[2] += norm_factor * ic

            gcn2_macro_acc[0] += norm_factor * pc2
            gcn2_macro_acc[1] += norm_factor * tc2
            gcn2_macro_acc[2] += norm_factor * ic2

            gcn_acc.append(curr_gcn_acc)
            gcn2_acc.append(curr_gcn2_acc)

            curr_idx += num_leafs
            bbox_idx += num_bbox
        
    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')

    tree_acc = np.array(tree_img_acc, dtype=np.int32)
    tree_class_acc = tree_acc[:, 1:].mean(0) * 100
    print_and_log('Tree Avg_class_Accuracy: %.2f%%' % (tree_class_acc.mean()), log_file)
    for c in range(1, n_classes):
        print_and_log('%s gcn-Acc: %.2f%%' % (model.class_names[c], tree_class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(tree_img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='tree')

    gcn_acc = np.array(gcn_acc, dtype=np.int32)
    gcn_class_acc = gcn_acc.mean(0) * 100
    print_and_log('GCN Avg_class_Accuracy: %.2f%%' % (gcn_class_acc.mean()), log_file)
    for c in range(1, n_classes):
        print_and_log('%s gcn-Acc: %.2f%%' % (model.class_names[c], gcn_class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(gcn_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn')

    gcn2_acc = np.array(gcn2_acc, dtype=np.int32)
    gcn2_class_acc = gcn2_acc.mean(0) * 100
    print_and_log('GCN + GRU Hidden Combined Avg_class_Accuracy: %.2f%%' % (gcn2_class_acc.mean()), log_file)
    for c in range(1, n_classes):
        print_and_log('%s gcn2-Acc: %.2f%%' % (model.class_names[c], gcn2_class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(gcn2_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn2')

    return img_acc, img_macro_acc/total_domain_num * 100, gcn_acc, gcn_macro_acc/total_domain_num*100

@torch.no_grad()
def evaluate_text_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
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

    n_classes = model.n_classes
    pre_rnn_acc = []
    pre_rnn_macro_acc = np.array([0.0, 0.0, 0.0])
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    gcn_acc = []
    gcn_macro_acc = np.array([0.0, 0.0, 0.0])
    total_domain_num = eval_loader.dataset.total_domain_num
    for i, (_, bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(eval_loader)):
        # if i == 30:
        #     break
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]

        encoded_features, pre_leaf_rnn_output = model.encode(bboxes.to(device), texts, chars, tag_paths, trees, rel_bboxes.to(device))
        output = model.decode(encoded_features)
        tree_output = output

        num_bboxes = model.num_bboxes_per_img(bboxes[:, 0])
        curr_idx = 0
        for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
            curr_img_acc = [0]
            curr_pre_rnn_acc = [0]
        
            output_segment = output[curr_idx:curr_idx + num_leafs]
            labels_segment = labels[curr_idx:curr_idx + num_leafs]
            (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                            match(output_segment, labels_segment, 2),
                            match(output_segment, labels_segment, 3)) 
            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic
            img_acc.append(curr_img_acc)

            pre_leaf_rnn_output_segment = pre_leaf_rnn_output[curr_idx:curr_idx + num_leafs] 
            (pc, tc, ic) = (match(pre_leaf_rnn_output_segment, labels_segment, 1),
                            match(pre_leaf_rnn_output_segment, labels_segment, 2),
                            match(pre_leaf_rnn_output_segment, labels_segment, 3)) 
            curr_pre_rnn_acc.append(pc)
            curr_pre_rnn_acc.append(tc)
            curr_pre_rnn_acc.append(ic)
            pre_rnn_macro_acc[0] += norm_factor * pc
            pre_rnn_macro_acc[1] += norm_factor * tc
            pre_rnn_macro_acc[2] += norm_factor * ic
            pre_rnn_acc.append(curr_pre_rnn_acc)

            curr_idx += num_leafs


        curr_idx = 0
        bbox_idx = 0
        for num_leafs, num_bbox, norm_factor in zip(leaf_nums, num_bboxes, norm_factors):
            curr_gcn_acc = []

            graph_bboxes = bboxes[bbox_idx:bbox_idx + num_leafs][:, 1:]
            graph_labels = labels[curr_idx:curr_idx + num_leafs]
            raw_predicted_prob = tree_output[curr_idx:curr_idx + num_leafs]
            node_features = encoded_features[curr_idx:curr_idx + num_leafs]
            graph, tmp_labels, select_indices = graph_constructor.construct(
                graph_bboxes, raw_predicted_prob, graph_labels, train=False, node_features=node_features, 
                return_select_indices=True)
            logits = model.gcn_forward(graph, node_features[select_indices])

            nc = model.num_candidates

            (pc, tc, ic) = (match(logits[:nc], tmp_labels[:nc], 1),
                            match(logits[nc:2*nc], tmp_labels[nc:2*nc], 2),
                            match(logits[2*nc:3*nc], tmp_labels[2*nc:3*nc], 3))

            curr_gcn_acc.append(pc)
            curr_gcn_acc.append(tc)
            curr_gcn_acc.append(ic)
            
            gcn_macro_acc[0] += norm_factor * pc
            gcn_macro_acc[1] += norm_factor * tc
            gcn_macro_acc[2] += norm_factor * ic

            gcn_acc.append(curr_gcn_acc)

            curr_idx += num_leafs
            bbox_idx += num_bbox

    pre_rnn_acc = np.array(pre_rnn_acc, dtype=np.int32)
    pre_rnn_class_acc = pre_rnn_acc[:,1:].mean(0) * 100
    print_and_log('Pre Leaf RNN Avg_class_Accuracy: %.2f%%' % (pre_rnn_class_acc.mean()), log_file)
    for c in range(1, n_classes):
        print_and_log('%s pre_rnn-Acc: %.2f%%' % (model.class_names[c], pre_rnn_class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(pre_rnn_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='pre_rnn')

    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')
    
    gcn_acc = np.array(gcn_acc, dtype=np.int32)
    gcn_class_acc = gcn_acc.mean(0) * 100
    print_and_log('GCN Avg_class_Accuracy: %.2f%%' % (gcn_class_acc.mean()), log_file)
    for c in range(1, n_classes):
        print_and_log('%s gcn-Acc: %.2f%%' % (model.class_names[c], gcn_class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(gcn_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn')

    return img_acc, img_macro_acc/total_domain_num * 100, gcn_acc, gcn_macro_acc/total_domain_num*100

@torch.no_grad()
def evaluate_vistext_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt', use_softmax=False):
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

    n_classes = model.n_classes
    # pre_rnn_acc = []
    # pre_rnn_macro_acc = np.array([0.0, 0.0, 0.0])
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    img_macro_acc = np.array([0.0, 0.0, 0.0])
    gcn_acc = []
    gcn_macro_acc = np.array([0.0, 0.0, 0.0])
    
    gcn_xpath_acc = []
    gcn_macro_xpath_acc = np.array([0.0, 0.0, 0.0])
    
    site_freq_xpath_dict = dict()
    label_dict = dict()
    tag_path_dict = dict()
    norm_factor_dict = dict()
    output_dict = dict()
    
    total_domain_num = eval_loader.dataset.total_domain_num
    pkl2domain = eval_loader.dataset.pkl2domain
    
    n_webpages = len(eval_loader.dataset)
    
    for i, (img_ids, images, leaf_bboxes, extended_bboxes, texts, chars, leaf_tags, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes) in enumerate(tqdm(eval_loader)):
        
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
            img_id = img_ids[batch_id]
            label_dict[img_id] = label_index
        for batch_id, norm_factor in enumerate(norm_factors):
            img_id = img_ids[batch_id]
            norm_factor_dict[img_id] = norm_factor
        tag_path_idx = 0
        tag_paths_segs = []
        for batch_id, num in enumerate(leaf_nums):
            img_id = img_ids[batch_id]
            tag_seg = tag_paths[tag_path_idx:tag_path_idx+num]
            tag_path_idx += num
            tag_path_dict[img_id] = tag_seg
            tag_paths_segs.append(tag_seg)

                    
        # if i == 30:
        #     break
        labels = labels.to(device) # [total_n_bboxes_in_batch, 4]

        # with torch.cuda.amp.autocast():
        encoded_features = model.encode(images.to(device), leaf_bboxes.to(device), extended_bboxes.to(device), texts, chars, leaf_tags.to(device), tag_paths, trees, rel_bboxes.to(device))
        output = model.decode(encoded_features)

        num_bboxes = model.num_bboxes_per_img(leaf_bboxes[:, 0])
        curr_idx = 0
        for i, (num_leafs, norm_factor, num_bbox) in enumerate(zip(leaf_nums, norm_factors, num_bboxes)):
            curr_img_acc = [0]
            # curr_pre_rnn_acc = [0]
        
            output_segment = output[curr_idx:curr_idx + num_leafs]
            labels_segment = labels[curr_idx:curr_idx + num_leafs]
            (pc, tc, ic) = (match(output_segment, labels_segment, 1),
                            match(output_segment, labels_segment, 2),
                            match(output_segment, labels_segment, 3)) 
            curr_img_acc.append(pc)
            curr_img_acc.append(tc)
            curr_img_acc.append(ic)
            img_macro_acc[0] += norm_factor * pc
            img_macro_acc[1] += norm_factor * tc
            img_macro_acc[2] += norm_factor * ic
            img_acc.append(curr_img_acc)

            # pre_leaf_rnn_output_segment = pre_leaf_rnn_output[curr_idx:curr_idx + num_leafs] 
            # (pc, tc, ic) = (match(pre_leaf_rnn_output_segment, labels_segment, 1),
            #                 match(pre_leaf_rnn_output_segment, labels_segment, 2),
            #                 match(pre_leaf_rnn_output_segment, labels_segment, 3)) 
            # curr_pre_rnn_acc.append(pc)
            # curr_pre_rnn_acc.append(tc)
            # curr_pre_rnn_acc.append(ic)
            # pre_rnn_macro_acc[0] += norm_factor * pc
            # pre_rnn_macro_acc[1] += norm_factor * tc
            # pre_rnn_macro_acc[2] += norm_factor * ic
            # pre_rnn_acc.append(curr_pre_rnn_acc)

            curr_idx += num_leafs

        
        curr_idx = 0
        bbox_idx = 0
        batch_output = []
        for j, (num_leafs, num_bbox, norm_factor) in enumerate(zip(leaf_nums, num_bboxes, norm_factors)):
            curr_gcn_acc = []

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

            curr_gcn_acc.append(pc)
            curr_gcn_acc.append(tc)
            curr_gcn_acc.append(ic)

            gcn_macro_acc[0] += norm_factor * pc
            gcn_macro_acc[1] += norm_factor * tc
            gcn_macro_acc[2] += norm_factor * ic

            gcn_acc.append(curr_gcn_acc)
            
            pred_idx_1 = select_indices[logits[:nc, 1].argmax()]
            pred_idx_2 = select_indices[logits[nc:2*nc, 2].argmax() + nc]
            pred_idx_3 = select_indices[logits[2*nc:3*nc, 3].argmax() + 2*nc]
            page_output = [pred_idx_1, pred_idx_2, pred_idx_3]
            img_id = img_ids[j]
            output_dict[img_id] = page_output
            batch_output.append(page_output)

            curr_idx += num_leafs
            bbox_idx += num_bbox
            
        for batch_id, (val, page_tag_paths) in enumerate(zip(batch_output, tag_paths_segs)):
            img_id = img_ids[batch_id]
            site = pkl2domain[img_id]
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
                    
    
    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # ignore class-0 (BG) accuracy
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(img_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc')
    
    gcn_acc = np.array(gcn_acc, dtype=np.int32)
    gcn_class_acc = gcn_acc.mean(0) * 100
    print_and_log('GCN Avg_class_Accuracy: %.2f%%' % (gcn_class_acc.mean()), log_file)
    for c in range(1, n_classes):
        print_and_log('%s gcn-Acc: %.2f%%' % (model.class_names[c], gcn_class_acc[c-1]), log_file)
    print_and_log('', log_file)

    print_pti_acc_stats(gcn_macro_acc, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='gcn')
    
    correct_macro, correct = xpath_voting(site_freq_xpath_dict, label_dict, tag_path_dict, norm_factor_dict, output_dict, pkl2domain)
    print_pti_acc_stats(correct, n_webpages, log_file)
    print_pti_acc_stats(correct_macro, total_domain_num, log_file, split_name=split_name, acc_name='Macro Acc', acc_modifier='xpath voting')

    return img_acc, img_macro_acc/total_domain_num * 100, gcn_acc, gcn_macro_acc/total_domain_num*100


def match_xpath(logits, labels, match_label, xpaths_segment):
    pred_idx = logits[:, match_label].argmax()
    match_label_indices = (labels == match_label).nonzero()
    # if len(match_label_indices) == 0:
    #     return False
    # print(xpaths_segment)
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
    #         print(most_freq_path, frequency)
            site_freq_xpaths[site].append(most_freq_path)

    xpath_vote_output = dict()
    page_idices = list(label_dict.keys())
    for page_idx in page_idices:
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
    #         print(freq_xpath, possible_nodes)
            xpath_vote_output[page_idx].append(possible_nodes)

    correct_macro = np.zeros(3)
    correct = np.zeros(3)
    for page_idx in page_idices:
        labels = label_dict[page_idx]
        xpath_outputs = xpath_vote_output[page_idx]
        norm = norm_factors[page_idx]
#         print("labels:", labels)
#         print("xpath_outputs:", xpath_outputs)
#         print("norm_factor:", norm)
        correct_macro[0] += norm*(labels[0] == xpath_outputs[0][0])
        correct_macro[1] += norm*(labels[1] == xpath_outputs[1][0])
        correct_macro[2] += norm*(labels[2] == xpath_outputs[2][0])
        correct[0] += (labels[0] == xpath_outputs[0][0])
        correct[1] += (labels[1] == xpath_outputs[1][0])
        correct[2] += (labels[2] == xpath_outputs[2][0])
#         print("correct_macro:",correct_macro)
    return correct_macro, correct

