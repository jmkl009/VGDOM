import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
import pandas as pd

from utils import pkl_load


class WebDataset(torchvision.datasets.VisionDataset):
    """
    Class to load train/val/test datasets
    """
    def __init__(self, root, img_ids, use_context, context_size, use_additional_features=False, sampling_fraction=1):
        """
        Args:
            root: directory where data is located
                Must contain imgs/x.png Image and corresponding bboxes/x.pkl Bounding Boxes files
                BBox file should have bboxes in pre-order and each row corresponds to [x,y,w,h,label]
            img_ids: list of img_names to consider
            use_context: whether to make use of context or not (boolean)
                if False, `context_indices` will be empty as it will not be used in training
            context_size: number of BBoxes before and after to consider as context (int)
                NOTE: this parameter is used only if use_context=True
            use_additional_features: whether to use additional features (default: False)
                if True, `root` directory must contain additional_features/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_features]
            sampling_fraction: randomly sample this many (float between 0 and 1) fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
                All samples of class > 0 are always taken
                NOTE: For val and test data, sampling_fraction SHOULD be 1 (no sampling)
        """
        super(WebDataset, self).__init__(root)
        
        self.ids = [int(id) for id in img_ids]
        self.use_context = use_context
        self.context_size = context_size
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.sampling_fraction = sampling_fraction

        self.imgs_paths = ['%s/imgs/%s.png' % (self.root, img_id) for img_id in self.ids]
        self.all_bboxes = [pkl_load('%s/bboxes/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        
        if use_additional_features:
            self.all_additional_tensor_features = [torch.Tensor(pkl_load('%s/additional_features/%s.pkl' % (self.root, img_id))) for img_id in self.ids]
            self.n_additional_features = len(self.all_additional_tensor_features[0][0])
        else:
            self.all_additional_tensor_features = [torch.empty(len(bboxes), 0) for bboxes in self.all_bboxes]
            self.n_additional_features = 0

        
        pkl_and_domain = pd.read_csv('%s/splits/webpage_info.csv' % self.root)
        pkl2domain = {pkl_name:domain_name for pkl_name, domain_name in zip(
                                                    pkl_and_domain['webpage_name'],
                                                    pkl_and_domain['domain']
                                                    )}
        domain2num = {}
        # print(pkl2domain)
        for pkl_id in self.ids:
            domain = pkl2domain[pkl_id]
            domain2num[domain] = domain2num.get(domain, 0) + 1
        ids = set(self.ids)
        self.pkl2coeff = {pkl_name:(1/domain2num[domain_name]) for pkl_name, domain_name in pkl2domain.items() if pkl_name in ids}
        self.total_domain_num = len(domain2num)

        print('Total domain num:', self.total_domain_num)
        # print(self.pkl2coeff)
        for sample_pkl, sample_coeff in self.pkl2coeff.items():
            sample_pkl2coeff = (sample_pkl, sample_coeff)
            break
        print('Sample:', pkl2domain[sample_pkl2coeff[0]], domain2num[pkl2domain[sample_pkl2coeff[0]]], sample_pkl2coeff[0], sample_pkl2coeff[1])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            img_id: name of image (string)
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_features: torch.Tensor of size [n_bbox, n_additional_features]
            context_indices: torch.LongTensor of size [n_bbox, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        img = Image.open(self.imgs_paths[index]).convert('RGB')
        img = self.img_transform(img)
        
        bboxes = self.all_bboxes[index]
        additional_features = self.all_additional_tensor_features[index]
        if self.sampling_fraction < 1: # preserve order, include all non-BG bboxes
            sampled_bbox_idxs = np.random.permutation(bboxes.shape[0])[:int(self.sampling_fraction * bboxes.shape[0])]
            indices = np.concatenate((np.where(bboxes[:,-1] != 0)[0], sampled_bbox_idxs))
            indices = np.unique(indices) # sort and remove duplicate non-BG boxes
            bboxes = bboxes[indices]
            additional_features = additional_features[indices]
        
        labels = torch.LongTensor(bboxes[:,-1])

        bboxes = torch.Tensor(bboxes[:,:-1])
        bboxes[:,2:] += bboxes[:,:2] # convert from [x,y,w,h] to [x1,y1,x2,y2]

        if self.use_context:
            context_indices = []
            for i in range(bboxes.shape[0]):
                context = list(range(max(0, i-self.context_size), i)) + list(range(i+1, min(bboxes.shape[0], i+self.context_size+1)))
                context_indices.append(context + [-1]*(2*self.context_size - len(context)))
            context_indices = torch.LongTensor(context_indices)
        else:
            context_indices = torch.empty((0, 0), dtype=torch.long)

        return img_id, img, bboxes, additional_features, context_indices, labels, self.pkl2coeff[self.ids[index]]

    def __len__(self):
        return len(self.ids)

########################## End of class `WebDataset` ##########################


def custom_collate_fn(batch):
    """
    Since all images might have different number of BBoxes, to use batch_size > 1,
    custom collate_fn has to be created that creates a batch
    
    Args:
        batch: list of N=`batch_size` tuples. Example [(img_id_1, img_1, bboxes_1, afs_1, ci_1, labels_1), ..., (img_id_N, img_N, bboxes_N, afs_N, ci_N, labels_N)]
    
    Returns:
        batch: contains img_ids, images, bboxes, context_indices, labels
            img_ids: names of images (string) to compute imgwise (webpagewise) and domainwise (macro) Accuracies
            images: torch.Tensor [N, 3, img_H, img_W]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_features: torch.Tensor of size [total_n_bboxes_in_batch, n_additional_features]
            context_indices: torch.LongTensor [total_n_bboxes_in_batch, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor [total_n_bboxes_in_batch]
    """
    img_ids, images, bboxes, additional_features, context_indices, labels, norm_factors = zip(*batch)
    # img_ids = (img_id_1, ..., img_id_N)
    # images = (img_1, ..., img_N) each element of size [3, img_H, img_W]
    # bboxes = (bboxes_1, ..., bboxes_N) each element of size [n_bboxes_in_image, 4]
    # additional_features = (additional_features_1, ..., additional_features_N) each element of size [n_bboxes_in_image, n_additional_features]
    # context_indices = (ci_1, ..., ci_N) each element of size [n_bboxes_in_image, 2*context_size]
    # labels = (labels_1, ..., labels_N) each element of size [n_bboxes_in_image]
    
    img_ids = np.array(img_ids)
    images = torch.stack(images, 0)
    
    bboxes_with_batch_index = []
    observed_bboxes = 0
    for i, bbox in enumerate(bboxes):
        batch_indices = torch.Tensor([i]*bbox.shape[0]).view(-1,1)
        bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
        context_indices[i][context_indices[i] != -1] += observed_bboxes
        observed_bboxes += bbox.shape[0]
    bboxes_with_batch_index = torch.cat(bboxes_with_batch_index)
    context_indices = torch.cat(context_indices)

    additional_features = torch.cat(additional_features)
    labels = torch.cat(labels)
    
    return img_ids, images, bboxes_with_batch_index, additional_features, context_indices, labels, list(norm_factors)

class WebTreeDataset(torchvision.datasets.VisionDataset):
    """
    Class to load train/val/test datasets
    """
    def __init__(self, root, img_ids, use_additional_features=False):
        super(WebTreeDataset, self).__init__(root)
        
        self.ids = [int(id) for id in img_ids]
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        self.imgs_paths = ['%s/imgs/%s.png' % (self.root, img_id) for img_id in self.ids]
        self.all_bboxes_and_labels = [pkl_load('%s/extended_pkls/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_rel_bboxes = [pkl_load('%s/rel_bboxes/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_bboxes = [bboxes_and_labels[0] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_labels = [bboxes_and_labels[1] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_leaf_nums = [bboxes_and_labels[2] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_trees = [pkl_load('%s/extended_pkl_trees/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        if use_additional_features:
            self.all_additional_tensor_features = [torch.Tensor(pkl_load('%s/additional_features/%s.pkl' % (self.root, img_id))) for img_id in self.ids]
            self.n_additional_features = len(self.all_additional_tensor_features[0][0])
        else:
            self.all_additional_tensor_features = [torch.empty(len(bboxes), 0) for bboxes in self.all_bboxes]
            self.n_additional_features = 0

        
        pkl_and_domain = pd.read_csv('%s/splits/webpage_info.csv' % self.root)
        pkl2domain = {pkl_name:domain_name for pkl_name, domain_name in zip(
                                                    pkl_and_domain['webpage_name'],
                                                    pkl_and_domain['domain']
                                                    )}
        domain2num = {}
        # print(pkl2domain)
        for pkl_id in self.ids:
            domain = pkl2domain[pkl_id]
            domain2num[domain] = domain2num.get(domain, 0) + 1
        ids = set(self.ids)
        self.pkl2coeff = {pkl_name:(1/domain2num[domain_name]) for pkl_name, domain_name in pkl2domain.items() if pkl_name in ids}
        self.total_domain_num = len(domain2num)

        print('Total domain num:', self.total_domain_num)
        # print(self.pkl2coeff)
        for sample_pkl, sample_coeff in self.pkl2coeff.items():
            sample_pkl2coeff = (sample_pkl, sample_coeff)
            break
        print('Sample:', pkl2domain[sample_pkl2coeff[0]], domain2num[pkl2domain[sample_pkl2coeff[0]]], sample_pkl2coeff[0], sample_pkl2coeff[1])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            img_id: name of image (string)
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_features: torch.Tensor of size [n_bbox, n_additional_features]
            context_indices: torch.LongTensor of size [n_bbox, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        img = Image.open(self.imgs_paths[index]).convert('RGB')
        img = self.img_transform(img)
        
        additional_features = self.all_additional_tensor_features[index]

        labels = torch.LongTensor(self.all_labels[index])

        bboxes = torch.Tensor(self.all_bboxes[index])
        bboxes[:,2:] += bboxes[:,:2] # convert from [x,y,w,h] to [x1,y1,x2,y2]

        return img_id, img, bboxes, additional_features, labels, self.all_trees[index], self.all_leaf_nums[index], self.pkl2coeff[self.ids[index]], torch.Tensor(self.all_rel_bboxes[index])

    def __len__(self):
        return len(self.ids)

def custom_collate_tree_fn(batch):

    img_ids, images, bboxes, additional_features, labels, trees, leaf_nums, norm_factors, rel_bboxes = zip(*batch)
    
    img_ids = np.array(img_ids)
    images = torch.stack(images, 0)
    
    bboxes_with_batch_index = []
    for i, bbox in enumerate(bboxes):
        batch_indices = torch.Tensor([i]*bbox.shape[0]).view(-1,1)
        bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
    bboxes_with_batch_index = torch.cat(bboxes_with_batch_index)

    additional_features = torch.cat(additional_features)
    labels = torch.cat(labels, dim=0)

    rel_bboxes = torch.cat(rel_bboxes, dim=0)
    
    return img_ids, images, bboxes_with_batch_index, additional_features, labels, list(trees), list(leaf_nums), list(norm_factors), rel_bboxes


def load_data(data_dir, train_img_ids, val_img_ids, test_img_ids, use_context, context_size, batch_size, use_additional_features=False, sampling_fraction=1, num_workers=4):
    """
    Args:
        data_dir: directory which contains imgs/x.png Image and corresponding bboxes/x.pkl BBox coordinates file
        train_img_ids: list of img_names to consider in train split
        val_img_ids: list of img_names to consider in val split
        test_img_ids: list of img_names to consider in test split
        use_context: whether to make use of context or not (boolean)
        context_size: number of BBoxes before and after to consider as context
        batch_size: size of batch in train_loader
        use_additional_features: whether to use additional features (default: False)
            if True, `root` directory must contain additional_features/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_features]
        sampling_fraction: randomly sample this many fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
            All samples of class > 0 are always taken
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    assert np.intersect1d(train_img_ids, val_img_ids).size == 0
    assert np.intersect1d(val_img_ids, test_img_ids).size == 0
    assert np.intersect1d(train_img_ids, test_img_ids).size == 0
    
    train_dataset = WebDataset(data_dir, train_img_ids, use_context, context_size, use_additional_features, sampling_fraction)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=custom_collate_fn, drop_last=False)

    val_dataset = WebDataset(data_dir, val_img_ids, use_context, context_size, use_additional_features, sampling_fraction=1)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_fn, drop_last=False)
    
    test_dataset = WebDataset(data_dir, test_img_ids, use_context, context_size, use_additional_features, sampling_fraction=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=num_workers,
                             collate_fn=custom_collate_fn, drop_last=False)
    
    print('No. of Images\t Train: %d\t Val: %d\t Test: %d\n' % ( len(train_dataset), len(val_dataset), len(test_dataset) ))
    
    return train_loader, val_loader, test_loader

def load_tree_data(data_dir, train_img_ids, val_img_ids, test_img_ids, batch_size, use_additional_features=False, num_workers=4):
    """
    Args:
        data_dir: directory which contains imgs/x.png Image and corresponding bboxes/x.pkl BBox coordinates file
        train_img_ids: list of img_names to consider in train split
        val_img_ids: list of img_names to consider in val split
        test_img_ids: list of img_names to consider in test split
        batch_size: size of batch in train_loader
        use_additional_features: whether to use additional features (default: False)
            if True, `root` directory must contain additional_features/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_features]
        sampling_fraction: randomly sample this many fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
            All samples of class > 0 are always taken
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    assert np.intersect1d(train_img_ids, val_img_ids).size == 0
    assert np.intersect1d(val_img_ids, test_img_ids).size == 0
    assert np.intersect1d(train_img_ids, test_img_ids).size == 0
    
    train_dataset = WebTreeDataset(data_dir, train_img_ids, use_additional_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=custom_collate_tree_fn, drop_last=False)

    val_dataset = WebTreeDataset(data_dir, val_img_ids, use_additional_features)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_tree_fn, drop_last=False)
    
    test_dataset = WebTreeDataset(data_dir, test_img_ids, use_additional_features)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=num_workers,
                             collate_fn=custom_collate_tree_fn, drop_last=False)
    
    print('No. of Images\t Train: %d\t Val: %d\t Test: %d\n' % ( len(train_dataset), len(val_dataset), len(test_dataset) ))
    
    return train_loader, val_loader, test_loader

class WebTextDataset(torchvision.datasets.VisionDataset):
    """
    Class to load train/val/test datasets
    """
    def __init__(self, root, img_ids):
        super(WebTextDataset, self).__init__(root)
        
        self.ids = [int(id) for id in img_ids]

        self.imgs_paths = ['%s/imgs/%s.png' % (self.root, img_id) for img_id in self.ids]
        self.all_rel_bboxes = [pkl_load('%s/rel_bboxes/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_bboxes_and_labels = [pkl_load('%s/extended_pkls/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        # self.all_bboxes = [bboxes_and_labels[0] for bboxes_and_labels in self.all_bboxes_and_labels]
        # self.orig_bboxes = [pkl_load('%s/bboxes/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_bboxes = [bboxes_and_labels[0] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_labels = [bboxes_and_labels[1] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_leaf_nums = [bboxes_and_labels[2] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_trees = [pkl_load('%s/extended_pkl_trees/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_text_and_tags = [pkl_load('%s/auxiliary_infos/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.word_dict, self.pretrained_word_vectors = pkl_load('%s/vocabs.pkl' % self.root)
        self.char_dict = pkl_load('%s/chars.pkl' % self.root)
        self.char_vocab_size = np.max(list(self.char_dict.values())) + 2 # Plus 2 if we want padding
        self.tag_dict = pkl_load('%s/tags.pkl' % self.root)
        self.tag_vocab_size = np.max(list(self.tag_dict.values())) + 2 # Plus 2 if we want padding
        # print(self.tag_dict)
        pkl_and_domain = pd.read_csv('%s/splits/webpage_info.csv' % self.root)
        pkl2domain = {pkl_name:domain_name for pkl_name, domain_name in zip(
                                                    pkl_and_domain['webpage_name'],
                                                    pkl_and_domain['domain']
                                                    )}
        domain2num = {}
        # print(pkl2domain)
        for pkl_id in self.ids:
            domain = pkl2domain[pkl_id]
            domain2num[domain] = domain2num.get(domain, 0) + 1
        ids = set(self.ids)
        self.pkl2coeff = {pkl_name:(1/domain2num[domain_name]) for pkl_name, domain_name in pkl2domain.items() if pkl_name in ids}
        self.total_domain_num = len(domain2num)

        print('Total domain num:', self.total_domain_num)
        # print(self.pkl2coeff)
        for sample_pkl, sample_coeff in self.pkl2coeff.items():
            sample_pkl2coeff = (sample_pkl, sample_coeff)
            break
        print('Sample:', pkl2domain[sample_pkl2coeff[0]], domain2num[pkl2domain[sample_pkl2coeff[0]]], sample_pkl2coeff[0], sample_pkl2coeff[1])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            img_id: name of image (string)
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_features: torch.Tensor of size [n_bbox, n_additional_features]
            context_indices: torch.LongTensor of size [n_bbox, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        texts, tag_paths = self.all_text_and_tags[index]
        # converted_labels = list(np.argmax(self.all_labels[index][:self.all_leaf_nums[index]],  axis=1))
        # print(tag_paths[converted_labels.index(1)])
        # print(tag_paths[converted_labels.index(2)])
        # if tag_paths[converted_labels.index(3)][-1] != 66 and tag_paths[converted_labels.index(3)][-1] != 47:
        #     print(tag_paths[converted_labels.index(3)][-1])
        # print(tag_paths[converted_labels.index(3)])
        numericalized_textss = [] # (node_num, sentence_len)
        numericalized_charss = [] # (node_num, sentence_len, word_len)
        for text in texts:
            numericalized_text = []
            numericalized_chars = []
            for word in text:
                numericalized_chars.append([self.char_dict[char] for char in word])
                numericalized_text.append(self.word_dict[word])
            numericalized_textss.append(numericalized_text)
            numericalized_charss.append(numericalized_chars)
            
        num_leafs = self.all_leaf_nums[index]
        labels = torch.LongTensor(self.all_labels[index][:num_leafs].argmax(axis=1))
        bboxes = torch.Tensor(self.all_bboxes[index][:num_leafs])
        bboxes[:,2:] += bboxes[:,:2] # convert from [x,y,w,h] to [x1,y1,x2,y2]

        return img_id, bboxes, numericalized_textss, numericalized_charss, tag_paths, labels, self.all_trees[index], num_leafs, self.pkl2coeff[self.ids[index]], torch.Tensor(self.all_rel_bboxes[index])

    def __len__(self):
        return len(self.ids)

def custom_collate_text_fn(batch):

    img_ids, bboxes, texts, chars, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes = zip(*batch)

    flattened_texts = [] # (batch_node_num, sentence_len)
    for page_texts in texts:
        flattened_texts += page_texts
    flattened_chars = [] # (batch_node_num, sentence_len, word_len)
    for page_chars in chars:
        flattened_chars += page_chars
    flattened_tag_paths = [] # (batch_node_num, tag_path_len)
    for page_tag_paths in tag_paths:
        flattened_tag_paths += page_tag_paths
    
    img_ids = np.array(img_ids)
    
    bboxes_with_batch_index = []
    for i, bbox in enumerate(bboxes):
        batch_indices = torch.Tensor([i]*bbox.shape[0]).view(-1,1)
        bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
    bboxes_with_batch_index = torch.cat(bboxes_with_batch_index)

    assert len(flattened_texts) == np.sum(leaf_nums) and len(flattened_texts) == len(flattened_chars)

    labels = torch.cat(labels, dim=0)
    rel_bboxes = torch.cat(rel_bboxes, dim=0)
    
    return img_ids, bboxes_with_batch_index, flattened_texts, flattened_chars, flattened_tag_paths, labels, list(trees), list(leaf_nums), list(norm_factors), rel_bboxes

def load_text_data(data_dir, train_img_ids, val_img_ids, test_img_ids, batch_size, num_workers=4):
    """
    Args:
        data_dir: directory which contains imgs/x.png Image and corresponding bboxes/x.pkl BBox coordinates file
        train_img_ids: list of img_names to consider in train split
        val_img_ids: list of img_names to consider in val split
        test_img_ids: list of img_names to consider in test split
        batch_size: size of batch in train_loader
        use_additional_features: whether to use additional features (default: False)
            if True, `root` directory must contain additional_features/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_features]
        sampling_fraction: randomly sample this many fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
            All samples of class > 0 are always taken
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    assert np.intersect1d(train_img_ids, val_img_ids).size == 0
    assert np.intersect1d(val_img_ids, test_img_ids).size == 0
    assert np.intersect1d(train_img_ids, test_img_ids).size == 0
    
    train_dataset = WebTextDataset(data_dir, train_img_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=custom_collate_text_fn, drop_last=False)

    val_dataset = WebTextDataset(data_dir, val_img_ids)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_text_fn, drop_last=False)
    
    test_dataset = WebTextDataset(data_dir, test_img_ids)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                             collate_fn=custom_collate_text_fn, drop_last=False)
    
    print('No. of Images\t Train: %d\t Val: %d\t Test: %d\n' % ( len(train_dataset), len(val_dataset), len(test_dataset) ))
    
    return train_loader, val_loader, test_loader

class VisTextDataset(torchvision.datasets.VisionDataset):
    """
    Class to load train/val/test datasets
    """
    def __init__(self, root, img_ids):
        super(VisTextDataset, self).__init__(root)
        
        self.ids = [int(id) for id in img_ids]
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        self.imgs_paths = ['%s/imgs/%s.png' % (self.root, img_id) for img_id in self.ids]
        self.all_rel_bboxes = [pkl_load('%s/rel_bboxes/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_bboxes_and_labels = [pkl_load('%s/extended_pkls/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_bboxes = [bboxes_and_labels[0] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_labels = [bboxes_and_labels[1] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_leaf_nums = [bboxes_and_labels[2] for bboxes_and_labels in self.all_bboxes_and_labels]
        self.all_trees = [pkl_load('%s/extended_pkl_trees/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.all_text_and_tags = [pkl_load('%s/auxiliary_infos/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        self.word_dict, self.pretrained_word_vectors = pkl_load('%s/vocabs.pkl' % self.root)
        self.char_dict = pkl_load('%s/chars.pkl' % self.root)
        self.char_vocab_size = np.max(list(self.char_dict.values())) + 2 # Plus 2 if we want padding
        self.tag_dict = pkl_load('%s/tags.pkl' % self.root)
        self.tag_vocab_size = np.max(list(self.tag_dict.values())) + 2 # Plus 2 if we want padding
        pkl_and_domain = pd.read_csv('%s/splits/webpage_info.csv' % self.root)
        pkl2domain = {pkl_name:domain_name for pkl_name, domain_name in zip(
                                                    pkl_and_domain['webpage_name'],
                                                    pkl_and_domain['domain']
                                                    )}
        self.pkl2domain = pkl2domain
        
        domain2num = {}
        # print(pkl2domain)
        for pkl_id in self.ids:
            domain = pkl2domain[pkl_id]
            domain2num[domain] = domain2num.get(domain, 0) + 1
        ids = set(self.ids)
        self.pkl2coeff = {pkl_name:(1/domain2num[domain_name]) for pkl_name, domain_name in pkl2domain.items() if pkl_name in ids}
        self.total_domain_num = len(domain2num)

        print('Total domain num:', self.total_domain_num)
        # print(self.pkl2coeff)
        for sample_pkl, sample_coeff in self.pkl2coeff.items():
            sample_pkl2coeff = (sample_pkl, sample_coeff)
            break
        print('Sample:', pkl2domain[sample_pkl2coeff[0]], domain2num[pkl2domain[sample_pkl2coeff[0]]], sample_pkl2coeff[0], sample_pkl2coeff[1])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            img_id: name of image (string)
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_features: torch.Tensor of size [n_bbox, n_additional_features]
            context_indices: torch.LongTensor of size [n_bbox, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        texts, tag_paths = self.all_text_and_tags[index]
        numericalized_textss = [] # (node_num, sentence_len)
        numericalized_charss = [] # (node_num, sentence_len, word_len)
        for text in texts:
            numericalized_text = []
            numericalized_chars = []
            # Cap number of words to 20
            for word in text:
                numericalized_chars.append([self.char_dict[char] for char in word])
                numericalized_text.append(self.word_dict[word])
            numericalized_textss.append(numericalized_text)
            numericalized_charss.append(numericalized_chars)
            
        numericalized_leaf_tags = torch.Tensor([path[-1] for path in tag_paths]).int()
            
        num_leafs = self.all_leaf_nums[index]
        labels = torch.LongTensor(self.all_labels[index][:num_leafs].argmax(axis=1))
        extended_bboxes = torch.Tensor(self.all_bboxes[index])
        leaf_bboxes = extended_bboxes[:num_leafs]
        extended_bboxes[:,2:] += extended_bboxes[:,:2] # convert from [x,y,w,h] to [x1,y1,x2,y2]

        img = Image.open(self.imgs_paths[index]).convert('RGB')
        img = self.img_transform(img)

        return img_id, img, leaf_bboxes, extended_bboxes, numericalized_textss, numericalized_charss, numericalized_leaf_tags, tag_paths, labels, self.all_trees[index], num_leafs, self.pkl2coeff[self.ids[index]], torch.Tensor(self.all_rel_bboxes[index])

    def __len__(self):
        return len(self.ids)

def custom_collate_vistext_fn(batch):

    img_ids, images, leaf_bboxes, extended_bboxes, texts, chars, leaf_tags, tag_paths, labels, trees, leaf_nums, norm_factors, rel_bboxes = zip(*batch)

    flattened_texts = [] # (batch_node_num, sentence_len)
    for page_texts in texts:
        flattened_texts += page_texts
    flattened_chars = [] # (batch_node_num, sentence_len, word_len)
    for page_chars in chars:
        flattened_chars += page_chars
    flattened_tag_paths = [] # (batch_node_num, tag_path_len)
    for page_tag_paths in tag_paths:
        flattened_tag_paths += page_tag_paths
        
    flattened_leaf_tags = torch.cat(leaf_tags, dim=0)
    
    img_ids = np.array(img_ids)
    images = torch.stack(images, 0)
    
    extended_bboxes_with_batch_index = []
    for i, bbox in enumerate(extended_bboxes):
        batch_indices = torch.Tensor([i]*bbox.shape[0]).view(-1,1)
        extended_bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
    extended_bboxes_with_batch_index = torch.cat(extended_bboxes_with_batch_index)

    leaf_bboxes_with_batch_index = []
    for i, bbox in enumerate(leaf_bboxes):
        batch_indices = torch.Tensor([i]*bbox.shape[0]).view(-1,1)
        leaf_bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
    leaf_bboxes_with_batch_index = torch.cat(leaf_bboxes_with_batch_index)

    assert len(flattened_texts) == np.sum(leaf_nums) and len(flattened_texts) == len(flattened_chars)

    labels = torch.cat(labels, dim=0)
    rel_bboxes = torch.cat(rel_bboxes, dim=0)
    
    return img_ids, images, leaf_bboxes_with_batch_index, extended_bboxes_with_batch_index, flattened_texts, flattened_chars, flattened_leaf_tags, flattened_tag_paths, labels, list(trees), list(leaf_nums), list(norm_factors), rel_bboxes

def load_vistext_data(data_dir, train_img_ids, val_img_ids, test_img_ids, batch_size, num_workers=4):
    """
    Args:
        data_dir: directory which contains imgs/x.png Image and corresponding bboxes/x.pkl BBox coordinates file
        train_img_ids: list of img_names to consider in train split
        val_img_ids: list of img_names to consider in val split
        test_img_ids: list of img_names to consider in test split
        batch_size: size of batch in train_loader
        use_additional_features: whether to use additional features (default: False)
            if True, `root` directory must contain additional_features/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_features]
        sampling_fraction: randomly sample this many fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
            All samples of class > 0 are always taken
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    assert np.intersect1d(train_img_ids, val_img_ids).size == 0
    assert np.intersect1d(val_img_ids, test_img_ids).size == 0
    assert np.intersect1d(train_img_ids, test_img_ids).size == 0
    
    train_dataset = VisTextDataset(data_dir, train_img_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=custom_collate_vistext_fn, drop_last=False)

    val_dataset = VisTextDataset(data_dir, val_img_ids)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_vistext_fn, drop_last=False)
    
    test_dataset = VisTextDataset(data_dir, test_img_ids)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                             collate_fn=custom_collate_vistext_fn, drop_last=False)
    
    print('No. of Images\t Train: %d\t Val: %d\t Test: %d\n' % ( len(train_dataset), len(val_dataset), len(test_dataset) ))
    
    return train_loader, val_loader, test_loader

import os
import dgl
from dgl import DGLGraph
from numba import jit

@jit(nopython=True, parallel = False, fastmath=True)
def min_max_bbox_x_vectorized(b1, b2):
    x1_min = b1[:, 0]
    x2_max = b2[:, 2]
    
    dist = x1_min - x2_max
    abs_dist = np.abs(dist)
    c11 = np.exp(-abs_dist/160).astype(np.float32)
    c12 = np.exp(-abs_dist/320).astype(np.float32)
    c13 = np.exp(-abs_dist/80).astype(np.float32)
    c14 = np.exp(-abs_dist/40).astype(np.float32)
    c21 = 1 - c11
    c22 = 1 - c12
    c23 = 1 - c13
    c24 = 1 - c14

    ret_rel = (dist > 0).astype(np.int32)
    return ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24), 1-ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24)

@jit(nopython=True, parallel = False, fastmath=True)
def max_min_bbox_x_vectorized(b1, b2):
    x1_max = b1[:, 2]
    x2_min = b2[:, 0]
    
    dist = x1_max - x2_min
    abs_dist = np.abs(dist)
    c11 = np.exp(-abs_dist/160).astype(np.float32)
    c12 = np.exp(-abs_dist/320).astype(np.float32)
    c13 = np.exp(-abs_dist/80).astype(np.float32)
    c14 = np.exp(-abs_dist/40).astype(np.float32)
    c21 = 1 - c11
    c22 = 1 - c12
    c23 = 1 - c13
    c24 = 1 - c14

    ret_rel = (dist > 0).astype(np.int32)
    return ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24), 1-ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24)

@jit(nopython=True, parallel = False, fastmath=True)
def min_max_bbox_y_vectorized(b1, b2):
    y1_min = b1[:, 1]
    y2_max = b2[:, 3]
    
    dist = y1_min - y2_max
    abs_dist = np.abs(dist)
    c11 = np.exp(-abs_dist/160).astype(np.float32)
    c12 = np.exp(-abs_dist/320).astype(np.float32)
    c13 = np.exp(-abs_dist/80).astype(np.float32)
    c14 = np.exp(-abs_dist/40).astype(np.float32)
    c21 = 1 - c11
    c22 = 1 - c12
    c23 = 1 - c13
    c24 = 1 - c14

    ret_rel = (dist > 0).astype(np.int32)
    return ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24), 1-ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24)

@jit(nopython=True, parallel = False, fastmath=True)
def max_min_bbox_y_vectorized(b1, b2):
    y1_max = b1[:, 3]
    y2_min = b2[:, 1]
    
    dist = y1_max - y2_min
    abs_dist = np.abs(dist)
    c11 = np.exp(-abs_dist/160).astype(np.float32)
    c12 = np.exp(-abs_dist/320).astype(np.float32)
    c13 = np.exp(-abs_dist/80).astype(np.float32)
    c14 = np.exp(-abs_dist/40).astype(np.float32)
    c21 = 1 - c11
    c22 = 1 - c12
    c23 = 1 - c13
    c24 = 1 - c14

    ret_rel = (dist > 0).astype(np.int32)
    return ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24), 1-ret_rel, (c11, c12, c13, c14, c21, c22, c23, c24)

@jit(nopython=True, parallel = False, fastmath=True)
def width_comparison_continuous_vectorized(b1, b2):
    width1 = b1[:, 2] - b1[:, 0]
    width2 = b2[:, 2] - b2[:, 0]
    ratio = width2 / width1

    ret_rel = (ratio > 1).astype(np.int32)
    inverse_ret_rel = 1 - ret_rel
    c1 = (ret_rel * (1/ratio) + inverse_ret_rel * ratio).astype(np.float32)
    c2 = 1 - c1 

    return ret_rel, (c1, c2), inverse_ret_rel, (c1, c2)

@jit(nopython=True, parallel = False, fastmath=True)
def height_comparison_continuous_vectorized(b1, b2):
    height1 = b1[:, 3] - b1[:, 1]
    height2 = b2[:, 3] - b2[:, 1]
    ratio = height2 / height1

    ret_rel = (ratio > 1).astype(np.int32)
    inverse_ret_rel = 1 - ret_rel
    c1 = (ret_rel * (1/ratio) + inverse_ret_rel * ratio).astype(np.float32)
    c2 = 1 - c1
    return ret_rel, (c1, c2), inverse_ret_rel, (c1, c2)

class GraphConstuctor:
    def __init__(self, num_candidates, device='cpu', model=None):
        self.vectorized_relations = [
            (min_max_bbox_x_vectorized, 2, 8), 
            (max_min_bbox_x_vectorized, 2, 8),
            (min_max_bbox_y_vectorized, 2, 8),
            (max_min_bbox_y_vectorized, 2, 8),
            (width_comparison_continuous_vectorized, 2, 2),
            (height_comparison_continuous_vectorized, 2, 2),
        ]
        self.continuous_relations = [
            # (center_relative_degree_30_with_radius, 12, 2),
            # (min_max_bbox_x, 2, 2),
            # (max_min_bbox_x, 2, 2),
            # (min_max_bbox_y, 2, 2),
            # (max_min_bbox_y, 2, 2),
            # (width_comparison_continuous, 2, 2),
            # (height_comparison_continuous, 2, 2),
        ]
        self.relations = [
            # (width_comparison, 3),
            # (height_comparison, 3)
        ]
        self.num_rels = np.sum([rel[1] for rel in self.relations]
                                +[rel[1] * rel[2] for rel in self.continuous_relations]
                                +[rel[1] * rel[2] for rel in self.vectorized_relations]) * 3 + 3
        self.num_classes = 3
        self.node_num = num_candidates * self.num_classes
        self.num_candidates = num_candidates
        self.device = device
        self._init_candidate_edges()
        self.model=model

    def _init_candidate_edges(self):
        self.candidate_edge_src = [[i] * (self.node_num - i - 1) for i in range(self.node_num - 1)]
        self.candidate_edge_src = np.concatenate(self.candidate_edge_src)
        self.candidate_edge_dst = [np.arange(i+1, self.node_num) for i in range(self.node_num)]
        self.candidate_edge_dst = np.concatenate(self.candidate_edge_dst)

    def construct(self, bboxes, raw_prob_output, labels=None, train=False, 
                node_features=None, return_select_indices=False):
        assert (not train) or (labels is not None)

        with torch.no_grad():
            top_k_predictions = torch.argsort(raw_prob_output, dim=0)[-self.num_candidates:]
            price_indices = top_k_predictions[:, 1]
            title_indices = top_k_predictions[:, 2]
            image_indices = top_k_predictions[:, 3]
            selected_indices = torch.cat((price_indices, title_indices, image_indices))

        bboxes = bboxes[selected_indices]
        raw_prob_output = raw_prob_output[selected_indices]
        labels = labels[selected_indices] 
        

        # Construct Nodes and edges
        edge_src = []
        edge_dst = []
        edge_type = []
        edge_norm = []
        curr_rel_type_offset = 0

        randomize_arr = np.arange(0, 3 * self.num_candidates)

            
        srcs_randomized = randomize_arr[self.candidate_edge_src]
        dsts_randomized = randomize_arr[self.candidate_edge_dst]
        b1s = bboxes[srcs_randomized].cpu().numpy()
        b2s = bboxes[dsts_randomized].cpu().numpy()

        # print(len(b1s), len(b2s))
        for rel, num_rel_type, num_bases in self.vectorized_relations:
            srcs = np.tile(self.candidate_edge_src, num_bases)
            dsts = np.tile(self.candidate_edge_dst, num_bases)
            edge_src += list(np.concatenate((srcs, dsts)))
            edge_dst += list(np.concatenate((dsts, srcs)))

            src_to_dst_rel_types, src_to_dst_coeffss, dst_to_src_rel_types, dst_to_src_coeffss = rel(b1s, b2s)
            edge_norm += list(np.concatenate((np.concatenate(src_to_dst_coeffss), np.concatenate(dst_to_src_coeffss))))
            # print('edge_norm:', len(edge_norm))
            
            num_rel_per_type = (self.num_rels - 3)//3
            src_type = srcs_randomized // self.num_candidates
            dst_type = dsts_randomized // self.num_candidates

            src_to_dst_edge_types = np.concatenate([num_rel_per_type * src_type + curr_rel_type_offset + num_rel_type * i + src_to_dst_rel_types for i in range(num_bases)])
            dst_to_src_edge_types = np.concatenate([num_rel_per_type * dst_type + curr_rel_type_offset + num_rel_type * i + dst_to_src_rel_types for i in range(num_bases)])

            edge_type += list(src_to_dst_edge_types)
            edge_type += list(dst_to_src_edge_types)
            # print('edge_type:', len(edge_type))
            # print(src_to_dst_edge_types)

            curr_rel_type_offset += num_rel_type * num_bases

        #Add Self-loops
        node_list = list(range(self.node_num))
        edge_src += node_list
        edge_dst += node_list
        edge_norm += [1] * self.node_num
        assert self.num_rels - 3 not in edge_type
        assert self.num_rels - 2 not in edge_type
        assert self.num_rels - 1 not in edge_type
        for src in node_list:
            src_type = src // self.num_candidates
            edge_type.append(self.num_rels - 3 + src_type)

        edge_src = torch.tensor(edge_src)
        edge_dst = torch.tensor(edge_dst)
        edge_type = torch.tensor(edge_type)
        edge_norm = torch.tensor(edge_norm).unsqueeze(1)

        g = DGLGraph((edge_src, edge_dst))
        g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
        # if self.model.splitted:
        #     g = g.to('cuda:1')
        # else:
        g = g.to(self.device)

        bboxes = bboxes[randomize_arr].to(self.device)

        bboxes[:, 2:] -= bboxes[:, :2] # convert to [top_left_x, top_left_y, width, height]
        # bboxes[:, 0] += np.random.uniform(1280) - 640
        # bboxes[:, 1] += np.random.uniform(1280) - 640
        bbox_asp_ratio = (bboxes[:, 2]/bboxes[:, 3]).view(-1, 1)
#         bbox_features = torch.cat((bboxes[:, 2:], bbox_asp_ratio), dim=1)
        # print(bboxes)
        bbox_features = torch.cat((bboxes, bbox_asp_ratio), dim=1)
        bbox_features = self.model.bbox_feat_encoder(bbox_features)
        # if self.model.splitted:
        #     bbox_features = bbox_features.to('cuda:1')


        # w = bboxes[:, 2] - bboxes[:, 0]
        # height = bboxes[:, 3] - bboxes[:, 1]
        # aspect_ratio = (height/w).view(-1, 1)
        h = [bbox_features]
        # g.ndata['h'] = raw_prob_output[randomize_arr] * 0.01
        tentative_label = torch.zeros((self.node_num, 4)).to(self.device)
        one_hot_indices = [1] * self.num_candidates + [2] * self.num_candidates + [3] * self.num_candidates
        tentative_label[np.arange(self.node_num), one_hot_indices] = 1
        # if self.model.splitted:
        #     tentative_label = tentative_label.to('cuda:1')
        h.append(tentative_label)
        h.append(raw_prob_output)

        if node_features is not None:
            h.append(self.model.intermediate_output_encoder(node_features[selected_indices][randomize_arr]))
            # h.append(node_features[selected_indices][randomize_arr])
        core_node_features = torch.cat(h, dim=1)


        # BG bboxes feature creation
        # bg_bboxes = original_bboxes[list(bg_indices)].to(self.device)
        # w = bg_bboxes[:, 2] - bg_bboxes[:, 0]
        # height = bg_bboxes[:, 3] - bg_bboxes[:, 1]
        # aspect_ratio = (height/w).view(-1, 1)
        # h = [(w / 320).view(-1, 1), (height / 320).view(-1, 1), aspect_ratio]
        # tentative_label = torch.zeros((len(bg_indices), 4)).to(self.device)
        # tentative_label[:, 0] = 1
        # h.append(tentative_label)
        # if node_features is not None:
        #     h.append(self.model.intermediate_output_encoder(node_features[list(bg_indices)]))
        # bg_node_features = torch.cat(h, dim=1)

        final_feature = core_node_features
        # print(final_feature.shape)
        g.ndata['h'] = final_feature
        # print(g.ndata['h'].shape)
        # print(raw_prob_output[randomize_arr])
        if labels is not None:      
            if return_select_indices:
                return  g, labels[randomize_arr], selected_indices
            return g, labels[randomize_arr]
        if return_select_indices:
            return g, selected_indices
        return g
    
class TreeGraphConstuctor:
    def __init__(self, num_candidates, device='cpu', model=None):
        self.vectorized_relations = [
            (min_max_bbox_x_vectorized, 2, 8), 
            (max_min_bbox_x_vectorized, 2, 8),
            (min_max_bbox_y_vectorized, 2, 8),
            (max_min_bbox_y_vectorized, 2, 8),
            (width_comparison_continuous_vectorized, 2, 2),
            (height_comparison_continuous_vectorized, 2, 2),
        ]
        self.num_rels = np.sum([rel[1] * rel[2] for rel in self.vectorized_relations]) + 1
        self.num_classes = 3
        
        self.device = device
        self.model=model
        

    def construct_dense(self, idx_offset, bboxes, raw_prob_output, labels=None, train=False, 
                node_features=None, return_select_indices=False):
        assert (not train) or (labels is not None)
        
        node_num = len(bboxes)
        
        candidate_edge_src = [[i] * (node_num - i - 1) for i in range(node_num - 1)]
        candidate_edge_src = np.concatenate(candidate_edge_src) + idx_offset
        candidate_edge_dst = [np.arange(i+1, node_num) for i in range(node_num)]
        candidate_edge_dst = np.concatenate(candidate_edge_dst) + idx_offset

        # Construct Nodes and edges
        edge_src = []
        edge_dst = []
        edge_type = []
        edge_norm = []
        curr_rel_type_offset = 0

        randomize_arr = np.arange(0, node_num) + idx_offset

            
        srcs_randomized = randomize_arr[candidate_edge_src]
        dsts_randomized = randomize_arr[candidate_edge_dst]
        b1s = bboxes[srcs_randomized].cpu().numpy()
        b2s = bboxes[dsts_randomized].cpu().numpy()

        # print(len(b1s), len(b2s))
        for rel, num_rel_type, num_bases in self.vectorized_relations:
            srcs = np.tile(self.candidate_edge_src, num_bases)
            dsts = np.tile(self.candidate_edge_dst, num_bases)
            edge_src += list(np.concatenate((srcs, dsts)))
            edge_dst += list(np.concatenate((dsts, srcs)))

            src_to_dst_rel_types, src_to_dst_coeffss, dst_to_src_rel_types, dst_to_src_coeffss = rel(b1s, b2s)
            edge_norm += list(np.concatenate((np.concatenate(src_to_dst_coeffss), np.concatenate(dst_to_src_coeffss))))
            # print('edge_norm:', len(edge_norm))
            
            src_to_dst_edge_types = np.concatenate([curr_rel_type_offset + num_rel_type * i + src_to_dst_rel_types for i in range(num_bases)])
            dst_to_src_edge_types = np.concatenate([curr_rel_type_offset + num_rel_type * i + dst_to_src_rel_types for i in range(num_bases)])

            edge_type += list(src_to_dst_edge_types)
            edge_type += list(dst_to_src_edge_types)
            # print('edge_type:', len(edge_type))
            # print(src_to_dst_edge_types)

            curr_rel_type_offset += num_rel_type * num_bases

        #Add Self-loops
        node_list = list(range(idx_offset, node_num+idx_offset))
        edge_src += node_list
        edge_dst += node_list
        edge_norm += [1] * node_num
        assert self.num_rels - 1 not in edge_type
        edge_type += [self.num_rels - 1] * node_num
        
        bboxes = bboxes[randomize_arr].to(self.device)

        bboxes[:, 2:] -= bboxes[:, :2] # convert to [top_left_x, top_left_y, width, height]
        bbox_asp_ratio = (bboxes[:, 2]/bboxes[:, 3]).view(-1, 1)
        bbox_features = torch.cat((bboxes, bbox_asp_ratio), dim=1)
        bbox_features = self.model.bbox_feat_encoder(bbox_features)
        h = [bbox_features]
        h.append(self.model.intermediate_output_encoder(node_features[selected_indices][randomize_arr]))
        
        
        return edge_src, edge_dst, edge_type, edge_norm
    
    def construct_dense_edges(self, idx_offset, bboxes, raw_prob_output, labels=None, train=False, 
                node_features=None, return_select_indices=False):
        
        edge_src = torch.tensor(edge_src)
        edge_dst = torch.tensor(edge_dst)
        edge_type = torch.tensor(edge_type)
        edge_norm = torch.tensor(edge_norm).unsqueeze(1)

        g = DGLGraph((edge_src, edge_dst))
        g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
        g = g.to(self.device)

        bboxes = bboxes[randomize_arr].to(self.device)

        bboxes[:, 2:] -= bboxes[:, :2] # convert to [top_left_x, top_left_y, width, height]
        # bboxes[:, 0] += np.random.uniform(1280) - 640
        # bboxes[:, 1] += np.random.uniform(1280) - 640
        bbox_asp_ratio = (bboxes[:, 2]/bboxes[:, 3]).view(-1, 1)
#         bbox_features = torch.cat((bboxes[:, 2:], bbox_asp_ratio), dim=1)
        # print(bboxes)
        bbox_features = torch.cat((bboxes, bbox_asp_ratio), dim=1)
        bbox_features = self.model.bbox_feat_encoder(bbox_features)


        # w = bboxes[:, 2] - bboxes[:, 0]
        # height = bboxes[:, 3] - bboxes[:, 1]
        # aspect_ratio = (height/w).view(-1, 1)
        h = [bbox_features]
        # g.ndata['h'] = raw_prob_output[randomize_arr] * 0.01
        tentative_label = torch.zeros((self.node_num, 4)).to(self.device)
        one_hot_indices = [1] * self.num_candidates + [2] * self.num_candidates + [3] * self.num_candidates
        tentative_label[np.arange(self.node_num), one_hot_indices] = 1
        if self.model.splitted:
            tentative_label = tentative_label.to('cuda:1')
        h.append(tentative_label)

        if node_features is not None:
            h.append(self.model.intermediate_output_encoder(node_features[selected_indices][randomize_arr]))
            # h.append(node_features[selected_indices][randomize_arr])
        core_node_features = torch.cat(h, dim=1)


        # BG bboxes feature creation
        # bg_bboxes = original_bboxes[list(bg_indices)].to(self.device)
        # w = bg_bboxes[:, 2] - bg_bboxes[:, 0]
        # height = bg_bboxes[:, 3] - bg_bboxes[:, 1]
        # aspect_ratio = (height/w).view(-1, 1)
        # h = [(w / 320).view(-1, 1), (height / 320).view(-1, 1), aspect_ratio]
        # tentative_label = torch.zeros((len(bg_indices), 4)).to(self.device)
        # tentative_label[:, 0] = 1
        # h.append(tentative_label)
        # if node_features is not None:
        #     h.append(self.model.intermediate_output_encoder(node_features[list(bg_indices)]))
        # bg_node_features = torch.cat(h, dim=1)

        final_feature = core_node_features
        # print(final_feature.shape)
        g.ndata['h'] = final_feature
        # print(g.ndata['h'].shape)
        # print(raw_prob_output[randomize_arr])
        if labels is not None:      
            if return_select_indices:
                return  g, labels[randomize_arr], selected_indices
            return g, labels[randomize_arr]
        if return_select_indices:
            return g, selected_indices
        return g