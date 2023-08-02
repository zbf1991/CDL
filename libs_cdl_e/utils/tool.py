import torch
import torch.nn.functional as F




def UpDate_labels(logit, logit_2nd, main_th, aux_th, main_label, aff_pred):
    logit_prob = F.softmax(logit, dim=1)
    logit_prob_max, logit_1st_label = torch.max(logit_prob, dim=1)
    logit_2nd_prob = F.softmax(logit_2nd, dim=1)
    logit_2nd_prob_max,logit_2nd_label  = torch.max(logit_2nd_prob, dim=1)

    main_region = (logit_prob_max > logit_2nd_prob_max)
    main_2nd_region = (logit_2nd_prob_max > logit_prob_max)
    main_2nd_region[logit_2nd_prob_max < aux_th] = False
    main_region[logit_prob_max < main_th] = False
    main_reliable_label = 255*torch.ones_like(main_label)
    main_reliable_label[main_region == True] = logit_1st_label[main_region == True]
    main_reliable_label[main_2nd_region == True] = logit_2nd_label[main_2nd_region == True]
    main_label[main_reliable_label !=255] = main_reliable_label[main_reliable_label!=255]

    low_main_region = torch.zeros_like(main_region).long()
    low_main_region[logit_prob_max < 0.5] = 1
    low_main_region_2nd = torch.zeros_like(main_region).long()
    low_main_region_2nd[logit_2nd_prob_max < 0.5] = 1
    low_region=torch.zeros_like(main_region).long()
    low_region[(low_main_region+low_main_region_2nd) == 2] = 1 #==2

    high_main_region = torch.zeros_like(main_region).long()
    high_main_region[logit_prob_max > 0.9] = 1
    high_main_region_2nd = torch.zeros_like(main_region).long()
    high_main_region_2nd[logit_2nd_prob_max > 0.9] = 1
    high_region=torch.zeros_like(main_region).long()
    high_region[(high_main_region+high_main_region_2nd) > 0.1] = 1
    high_region[main_label==255] = 0
    main_label_3rd = main_reliable_label.clone()

    m_b, m_h, m_w = main_label_3rd.shape
    high_region = high_region.view(m_b, m_h * m_w, 1).float()
    aff_high_region = high_region.bmm(torch.ones_like(high_region.transpose(2, 1))).long()

    exp_aff_pred = torch.exp(aff_pred-1)
    exp_aff_pred_sum = torch.sum(exp_aff_pred*aff_high_region, dim=1, keepdim=True) #for 3-d, dim=1 for column
    nor_aff_pred = (exp_aff_pred*aff_high_region) / (exp_aff_pred_sum+1e-5)

    l_b, l_c, l_h, l_w = logit.size()
    logit_flatten = logit.resize(l_b, l_c, l_h*l_w)
    logit_2nd_flatten = logit_2nd.resize(l_b, l_c, l_h*l_w)

    aff_pred_logit = 1*logit_flatten.bmm(nor_aff_pred) + 1*logit_2nd_flatten.bmm(nor_aff_pred)
    new_pred_main = 0.5*(logit_flatten + logit_2nd_flatten) + 0.5*aff_pred_logit
    new_pred_main = new_pred_main.resize(l_b, l_c, l_h, l_w)
    new_pred_main = torch.argmax(new_pred_main, dim=1)
    main_label[low_region == 1] = new_pred_main[low_region == 1]
    main_label_2nd = main_label.clone()


    return main_label, main_label_2nd, main_label_3rd



def UpDate_labels_aff_pred_label(logit, logit_2nd, main_th, aux_th, main_label, aff_pred_main_label):
    logit_prob = F.softmax(logit, dim=1)
    logit_prob_max = torch.max(logit_prob, dim=1)[0]
    logit_2nd_prob = F.softmax(logit_2nd, dim=1)
    logit_2nd_prob_max = torch.max(logit_2nd_prob, dim=1)[0]
    logit_2nd_label = torch.argmax(logit_2nd_prob, dim=1)
    logit_1st_label = torch.argmax(logit_prob, dim=1)

    main_region = (logit_prob_max > logit_2nd_prob_max)
    main_2nd_region = (logit_2nd_prob_max > logit_prob_max)
    main_2nd_region[logit_2nd_prob_max < aux_th] = False
    main_region[logit_prob_max < main_th] = False
    main_label[main_region == True] = logit_1st_label[main_region == True]
    main_label[main_2nd_region == True] = logit_2nd_label[main_2nd_region == True]

    low_main_region = torch.zeros_like(main_region)
    low_main_region[logit_prob_max < 0.5] = 1
    low_main_region_2nd = torch.zeros_like(main_region)
    low_main_region_2nd[logit_2nd_prob_max < 0.5] = 1
    low_region=torch.zeros_like(main_region)
    low_region[(low_main_region+low_main_region_2nd) == 2] = 1

    high_main_region = torch.zeros_like(main_region)
    high_main_region[logit_prob_max > 0.9] = 1
    high_main_region_2nd = torch.zeros_like(main_region)
    high_main_region_2nd[logit_2nd_prob_max > 0.9] = 1
    high_region=torch.zeros_like(main_region)
    high_region[(high_main_region+high_main_region_2nd) > 0.1] = 1

    l_b, l_c, l_h, l_w = logit.size()
    aff_pred_label = []
    for aff_i, aff_pred_single_label in enumerate(aff_pred_main_label):
        aff_pred_single_label = F.interpolate(aff_pred_single_label, ([l_h, l_w]), mode='bilinear')
        aff_pred_label.append(aff_pred_single_label)

    aff_pred_label = torch.max(torch.stack(aff_pred_label), dim=0)[0]
    aff_pred_label = aff_pred_label.resize(l_b, l_c, l_h, l_w)
    new_pred_main = 0.5*logit + 0.5*logit_2nd + 1*aff_pred_label
    new_pred_main = torch.argmax(new_pred_main, dim=1)

    main_label[low_region == 1] = new_pred_main[low_region == 1]
    main_label_2nd = main_label.clone()

    return main_label, main_label_2nd



def compute_affinity_loss(fts_distance, labels, nclass=21):
    labels_t=labels.clone()
    labels_t[labels==255] = nclass
    one_hot_labels = F.one_hot(labels_t, nclass+1)
    one_hot_labels = one_hot_labels[:,:,:, :nclass]
    lb, lh, lw, lc = one_hot_labels.shape
    one_hot_labels = one_hot_labels.view(lb, lh*lw, lc).float()
    aff_labels = one_hot_labels.bmm(one_hot_labels.transpose(2, 1))
    valid_region = torch.ones_like(labels)
    valid_region[labels==255] = 0
    valid_region = valid_region.resize(lb, lh*lw, 1).float()
    valid_aff_region = valid_region.bmm(valid_region.transpose(2, 1))

    # pos_aff_region = valid_aff_region.clone()
    # pos_aff_region[aff_labels==0] = 0
    #
    # neg_aff_region = valid_aff_region.clone()
    # neg_aff_region[aff_labels==1] = 0
    pos_aff_region = aff_labels*valid_aff_region
    neg_aff_region = (1-aff_labels)*valid_aff_region

    aff_pos_loss = -torch.sum(pos_aff_region*torch.log(fts_distance+1e-5)) / (torch.sum(pos_aff_region)+1e-5)
    aff_neg_loss = -torch.sum(neg_aff_region*torch.log(1-fts_distance+1e-5)) / (torch.sum(neg_aff_region)+1e-5)

    aff_loss = 1*aff_pos_loss+2*aff_neg_loss

    return aff_loss



def UpDate_labels_test(logit, logit_2nd, main_th, aux_th, main_label, aff_pred):
    logit_prob = F.softmax(logit, dim=1)
    logit_prob_max = torch.max(logit_prob, dim=1)[0]
    logit_2nd_prob = F.softmax(logit_2nd, dim=1)
    logit_2nd_prob_max = torch.max(logit_2nd_prob, dim=1)[0]
    logit_2nd_label = torch.argmax(logit_2nd_prob, dim=1)
    logit_1st_label = torch.argmax(logit_prob, dim=1)

    main_region = (logit_prob_max > logit_2nd_prob_max)
    main_2nd_region = (logit_2nd_prob_max > logit_prob_max)
    main_2nd_region[logit_2nd_prob_max < aux_th] = False
    main_region[logit_prob_max < main_th] = False
    main_reliable_label = 255*torch.ones_like(main_label)
    main_reliable_label[main_region == True] = logit_1st_label[main_region == True]
    main_reliable_label[main_2nd_region == True] = logit_2nd_label[main_2nd_region == True]
    # main_label[main_region == True] = logit_1st_label[main_region == True]
    # main_label[main_2nd_region == True] = logit_2nd_label[main_2nd_region == True]
    main_label[main_reliable_label !=255] = main_reliable_label[main_reliable_label!=255]

    low_main_region = torch.zeros_like(main_region).long()
    low_main_region[logit_prob_max < 0.5] = 1
    low_main_region_2nd = torch.zeros_like(main_region).long()
    low_main_region_2nd[logit_2nd_prob_max < 0.5] = 1
    low_region=torch.zeros_like(main_region).long()
    low_region[(low_main_region+low_main_region_2nd) == 2] = 1 #==2

    high_main_region = torch.zeros_like(main_region).long()
    high_main_region[logit_prob_max > 0.9] = 1
    high_main_region_2nd = torch.zeros_like(main_region).long()
    high_main_region_2nd[logit_2nd_prob_max > 0.9] = 1
    high_region=torch.zeros_like(main_region).long()
    high_region[(high_main_region+high_main_region_2nd) > 0.1] = 1
    high_region[main_label==255] = 0


    return main_label

