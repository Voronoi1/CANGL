import copy
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from gradient_lib import *
from federated_learning import *
from Main_framework_utils import *
from Main_framework_lib2 import set_agr_for_server

import global_var

device = global_var.get_device()


"""
AGR-aware stealth poisoning utilities.

This file is intentionally separate from Main_framework_lib2.py.  It keeps the
same public epoch function name so a poisoning script can import this module
instead of Main_framework_lib2 when running the AGR-aware stealth variant.
"""


def _clone_detached_gradient(gradient):
    return [grad.detach().clone() for grad in gradient]


def _ordered_malicious_indices(poi_index):
    return [index for index, flag in enumerate(poi_index) if flag == 1]


def _build_full_gradient_list(client_num, mal_client_idx, benign_gradient_by_idx, malicious_gradients):
    full_gradient_list = []
    mal_position = {client_idx: pos for pos, client_idx in enumerate(mal_client_idx)}
    for client_idx in range(client_num):
        if client_idx in mal_position:
            full_gradient_list.append(malicious_gradients[mal_position[client_idx]])
        else:
            full_gradient_list.append(benign_gradient_by_idx[client_idx])
    return full_gradient_list


def _mean_all_client_gradients(gradient_list):
    size = len(gradient_list[0])
    mean_gradient = []
    for layer_idx in range(size):
        layer_grads = [gradient[layer_idx] for gradient in gradient_list]
        mean_gradient.append(torch.mean(torch.stack(layer_grads), dim=0))
    return mean_gradient


def _weighted_mean_all_client_gradients(benign_gradient_list, benign_weights, malicious_gradients):
    total_weight = float(sum(benign_weights) + len(malicious_gradients))
    if len(benign_gradient_list) > 0:
        size = len(benign_gradient_list[0])
    else:
        size = len(malicious_gradients[0])
    mean_gradient = []
    for layer_idx in range(size):
        layer_sum = None
        for benign_gradient, weight in zip(benign_gradient_list, benign_weights):
            weighted_layer = benign_gradient[layer_idx] * float(weight)
            if layer_sum is None:
                layer_sum = weighted_layer
            else:
                layer_sum = layer_sum + weighted_layer
        for malicious_gradient in malicious_gradients:
            if layer_sum is None:
                layer_sum = malicious_gradient[layer_idx]
            else:
                layer_sum = layer_sum + malicious_gradient[layer_idx]
        mean_gradient.append(layer_sum / total_weight)
    return mean_gradient


def _weighted_mean_gradients(gradient_list, weights=None):
    if len(gradient_list) == 0:
        return []
    if weights is None:
        weights = [1 for _ in gradient_list]

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Gradient weights must sum to a positive value.")

    size = len(gradient_list[0])
    mean_gradient = []
    for layer_idx in range(size):
        layer_sum = None
        for gradient, weight in zip(gradient_list, weights):
            weighted_layer = gradient[layer_idx] * float(weight)
            if layer_sum is None:
                layer_sum = weighted_layer
            else:
                layer_sum = layer_sum + weighted_layer
        mean_gradient.append(layer_sum / total_weight)
    return mean_gradient


def _gradient_l2_distance(gradient, center):
    distance_square = 0.0
    for grad, center_grad in zip(gradient, center):
        distance_square = distance_square + ((grad - center_grad) ** 2).sum()
    return torch.sqrt(distance_square + 1e-12)


def _stealth_hinge_loss(malicious_gradient, benign_center, epsilon):
    if len(benign_center) == 0:
        return malicious_gradient[0].sum() * 0.0
    distance = _gradient_l2_distance(malicious_gradient, benign_center)
    hinge = F.relu(distance - float(epsilon))
    return hinge ** 2


def _mean_stealth_hinge_loss(malicious_gradients, benign_center, epsilon):
    losses = [
        _stealth_hinge_loss(malicious_gradient, benign_center, epsilon)
        for malicious_gradient in malicious_gradients
    ]
    return torch.mean(torch.stack(losses))


def _mean_malicious_distance(malicious_gradients, benign_center):
    if len(benign_center) == 0:
        return malicious_gradients[0][0].sum() * 0.0
    distances = [
        _gradient_l2_distance(malicious_gradient, benign_center)
        for malicious_gradient in malicious_gradients
    ]
    return torch.mean(torch.stack(distances))


def _build_surrogate_benign_gradients(client_num, poi_index, ref_benign_gradient_list):
    """Represent benign slots with unique attacker-side clean gradients and counts."""
    if len(ref_benign_gradient_list) == 0:
        return [], []

    ref_num = len(ref_benign_gradient_list)
    benign_gradient_list = list(ref_benign_gradient_list)
    benign_weights = [0 for _ in range(ref_num)]
    benign_count = 0
    for client_idx in range(client_num):
        if poi_index[client_idx] == 0:
            benign_weights[benign_count % ref_num] += 1
            benign_count += 1

    return benign_gradient_list, benign_weights


def _build_shared_malicious_gradients(
    gradient_poi_unit, ref_benign_gradient, gamma, belta, malicious_num
):
    shared_gradient = [
        gamma * poi_grad + belta * ref_grad
        for poi_grad, ref_grad in zip(gradient_poi_unit, ref_benign_gradient)
    ]
    return [shared_gradient for _ in range(malicious_num)]


def _replicate_malicious_gradient(gradient, malicious_num):
    return [_clone_detached_gradient(gradient) for _ in range(malicious_num)]


def optimize_mal_by_agr_aware(
    server,
    benign_gradient_list,
    gradient_AGR,
    gradient_poi,
    ref_benign_gradient_list,
    poi_index,
    logger,
    benign_weights=None,
    optimization_round=100,
    coeff=[1, 0, 0, 0],
    threshold=[3e+2, 1e3, 3e+4],
    gamma_bound=[-2, 2],
    init_value=[1, 1],
    lr=1e-4,
    stealth_coeff=1e-4,
    stealth_epsilon=None,
    unit_dir_opt=False,
    opt_parameter="gamma",
    ifprint=True,
    **kwargs
):
    """
    Optimize the shared malicious update with the same scheme as
    Main_framework_lib2.optimize_mal_by_unit_scale_with_ref (two scalars gamma /
    belta, the same 4-term coefficient loss, the same guard branches and the
    same stopping rules), plus a weak stealth hinge penalty:

        stealth_loss = relu(distance(delta_mal, benign_center) - epsilon)^2

    `stealth_coeff` defaults to 1e-4.  When `stealth_epsilon` is not provided,
    sqrt(threshold[1]) is used because the existing threshold is expressed in
    the squared-distance scale used by gradient_distance_loss.

    Only three things stay AGR-aware:

    1. All malicious clients upload one identical update, parameterized as

           delta^mal = gamma * unit(gradient_poi) + belta * mean(ref_j)

       where ref_j is the normal local gradient calculated by malicious client j.
    2. The losses are measured on the aggregation output rather than on
       delta^mal alone, i.e.

           min || weighted_mean({delta_ben} U {delta_mal}) - gradient_AGR ||^2

    3. The benign slots are represented by weighted surrogate benign gradients
       built by _build_surrogate_benign_gradients (attacker-side clean gradients
       replicated round-robin with integer weights).
    """

    mal_client_idx = _ordered_malicious_indices(poi_index)
    malicious_num = len(mal_client_idx)

    if malicious_num == 0:
        return []

    benign_gradient_list = [
        _clone_detached_gradient(gradient) for gradient in benign_gradient_list
    ]
    if benign_weights is None:
        benign_weights = [1 for _ in benign_gradient_list]

    ref_benign_gradient_list = [
        _clone_detached_gradient(ref_gradient) for ref_gradient in ref_benign_gradient_list
    ]
    ref_benign_gradient = cal_gradient_mean(ref_benign_gradient_list)
    benign_center = _weighted_mean_gradients(benign_gradient_list, benign_weights)
    if len(benign_center) == 0:
        benign_center = ref_benign_gradient
    if stealth_epsilon is None:
        stealth_epsilon = float(threshold[1]) ** 0.5

    initial_stealth_loss = _mean_stealth_hinge_loss(
        [gradient_poi for _ in range(malicious_num)], benign_center, stealth_epsilon
    )
    if (
        gradient_distance_loss(gradient_AGR, gradient_poi) < threshold[0]
        and initial_stealth_loss.detach().item() == 0.0
    ):
        return _replicate_malicious_gradient(gradient_poi, malicious_num)
    else:
        gradient_poi_unit = copy_gradient_unit(gradient_poi)
        gamma = torch.Tensor([init_value[0]]).to(device).requires_grad_(True)
        belta = torch.Tensor([init_value[1]]).to(device).requires_grad_(True)

        opt1 = optim.SGD([gamma], lr=lr, momentum=0.8, weight_decay=5e-4)
        opt2 = optim.SGD([belta], lr=lr, momentum=0.8, weight_decay=5e-4)
        opt3 = optim.SGD([gamma, belta], lr=lr, momentum=0.8, weight_decay=5e-4)

        t = 0
        ref_benign_dis_loss = gradient_distance_loss(
                    ref_benign_gradient, gradient_AGR)

        if ref_benign_dis_loss > threshold[2]:
            if ifprint:
                print("Too Large Ref  Benign Gradient!")
            logger.info("Too Large Ref  Benign Gradient!")
            malicious_grad_scale = [gamma * grad for grad in gradient_poi_unit]

            if unit_dir_opt:
                malicious_grad_scale = cal_gradient_unit(malicious_grad_scale)
            return _replicate_malicious_gradient(malicious_grad_scale, malicious_num)

        if torch.any(torch.isnan(ref_benign_gradient[0].data)):
            if ifprint:
                print("Nan Ref Benign Gradient!")
            logger.info("Nan Ref Benign Gradient!")
            malicious_grad_scale = [gamma * grad for grad in gradient_poi_unit]

            if unit_dir_opt:
                malicious_grad_scale = cal_gradient_unit(malicious_grad_scale)
            return _replicate_malicious_gradient(malicious_grad_scale, malicious_num)

        if cal_gradient_if_zero(gradient_AGR):
            if ifprint:
                print("Zero AGR Gradient!")
            logger.info("Zero AGR Gradient!")
            opt_gradient = [init_value[0]*grad for grad in gradient_poi_unit]
            aggregation_gradient = _weighted_mean_all_client_gradients(
                benign_gradient_list,
                benign_weights,
                [opt_gradient for _ in range(malicious_num)],
            )
            benign_dis_loss = gradient_distance_loss(
                aggregation_gradient, gradient_AGR)
            mal_dis_loss = gradient_distance_loss(
                aggregation_gradient, gradient_poi)
            stealth_loss = _mean_stealth_hinge_loss(
                [opt_gradient for _ in range(malicious_num)],
                benign_center,
                stealth_epsilon,
            )
            stealth_distance = _mean_malicious_distance(
                [opt_gradient for _ in range(malicious_num)], benign_center
            )

            if ifprint:
                print("--------------------- Final Malicious Update --------------------")
                print("Previous AGR Gradient Distance Loss:", benign_dis_loss.item())
                print("Poisoning Gradient Distance Loss:", mal_dis_loss.item())
                print("Stealth Distance:", stealth_distance.item())
                print("Stealth Loss:", stealth_loss.item())
                print("---------------------------------------------------------------------")

            logger.info("--------------------- Final Malicious Update --------------------")
            logger.info("Previous AGR Gradient Distance Loss:{:.6f}".format(benign_dis_loss.item()))
            logger.info("Poisoning Gradient Distance Loss:{:.6f}".format(mal_dis_loss.item()))
            logger.info("Stealth Distance:{:.6f}".format(stealth_distance.item()))
            logger.info("Stealth Loss:{:.6f}".format(stealth_loss.item()))
            logger.info("----------------------------------------------------------------")

            return _replicate_malicious_gradient(opt_gradient, malicious_num)

        #print("Malicious Update Optimizing")
        while True:

            malicious_gradients = _build_shared_malicious_gradients(
                gradient_poi_unit, ref_benign_gradient, gamma, belta, malicious_num)

            if unit_dir_opt:
                malicious_gradients = [cal_gradient_unit(malicious_gradient)
                                       for malicious_gradient in malicious_gradients]
                gradient_AGR = cal_gradient_unit(gradient_AGR)

            aggregation_gradient = _weighted_mean_all_client_gradients(
                benign_gradient_list, benign_weights, malicious_gradients)

            benign_dis_loss = gradient_distance_loss(
                aggregation_gradient, gradient_AGR)
            mal_dis_loss = gradient_distance_loss(
                aggregation_gradient, gradient_poi)
            benign_similarity_loss = gradient_cos_similarity_loss(
                aggregation_gradient, gradient_AGR)
            mal_similarity_loss = gradient_cos_similarity_loss(
                aggregation_gradient, gradient_poi)

            agr_loss = coeff[0] * benign_dis_loss+coeff[1] * benign_similarity_loss + \
                    coeff[2] * mal_dis_loss+coeff[3] * mal_similarity_loss
            stealth_loss = _mean_stealth_hinge_loss(
                malicious_gradients, benign_center, stealth_epsilon
            )
            stealth_distance = _mean_malicious_distance(
                malicious_gradients, benign_center
            )
            loss = agr_loss + float(stealth_coeff) * stealth_loss

            if  loss < threshold[0]:
                # print("Break!!!")
                break

            if torch.any(torch.isnan(gamma.data)) or torch.any(torch.isnan(belta.data)):
                gamma.data = torch.Tensor([init_value[0]]).to(device).requires_grad_(True)
                belta.data = torch.Tensor([0]).to(device).requires_grad_(True)
                if ifprint:
                    print("Zero Coeff!")
                logger.info("Zero Coeff!")
                continue

            if benign_dis_loss > threshold[1]:
                gamma.data = torch.Tensor([init_value[0]]).to(device).requires_grad_(True)
                belta.data = torch.Tensor([0]).to(device).requires_grad_(True)
                if ifprint:
                    print("Large Previous AGR Gradient Distance!")
                    print(benign_dis_loss)
                logger.info("Large Previous AGR Gradient Distance!")
                logger.info(benign_dis_loss)
                break

            if t > optimization_round:
                if ifprint:
                    print("Out!!!")
                logger.info("Out!!!")
                break

            if opt_parameter == "gamma":
                opt1.zero_grad()
                loss.backward()
                opt1.step()
            elif opt_parameter == "belta":
                opt2.zero_grad()
                loss.backward()
                opt2.step()
            elif opt_parameter == "both":
                opt3.zero_grad()
                loss.backward()
                opt3.step()

            t += 1

        if ifprint:
            print("--------------------- Final Malicious Update --------------------")
            print("Previous AGR Gradient Distance Loss:", benign_dis_loss.item())
            print("Previous AGR Gradient Similarity Loss:", benign_similarity_loss.item())
            print("Poisoning Gradient Distance Loss:", mal_dis_loss.item())
            print("Poisoning Gradient Similarity Loss:", mal_similarity_loss.item())
            print("AGR Loss:", agr_loss.item())
            print("Stealth Distance:", stealth_distance.item())
            print("Stealth Epsilon:", stealth_epsilon)
            print("Stealth Loss:", stealth_loss.item())
            print("Stealth Coeff:", stealth_coeff)
            print("Total Loss:", loss.item())
            print("---------------------------------------------------------------------")

        if torch.isnan(benign_dis_loss):
            logger.info(malicious_gradients[0][0][0])
            logger.info("============================================================================================================")
            logger.info(gradient_poi_unit[0][0])
            logger.info("============================================================================================================")
            logger.info(ref_benign_gradient[0][0])
            logger.info("============================================================================================================")
            logger.info(gamma)
            logger.info(belta)

        logger.info("--------------------- Final Malicious Update --------------------")
        logger.info("Previous AGR Gradient Distance Loss:{:.6f}".format(benign_dis_loss.item()))
        logger.info("Previous AGR Gradient Similarity Loss:{:.6f}".format(benign_similarity_loss.item()))
        logger.info("Poisoning Gradient Loss:{:.6f}".format(mal_dis_loss.item()))
        logger.info("Poisoning Gradient Similarity:{:.6f}".format(mal_similarity_loss.item()))
        logger.info("AGR Loss:{:.6f}".format(agr_loss.item()))
        logger.info("Stealth Distance:{:.6f}".format(stealth_distance.item()))
        logger.info("Stealth Epsilon:{:.6f}".format(float(stealth_epsilon)))
        logger.info("Stealth Loss:{:.6f}".format(stealth_loss.item()))
        logger.info("Stealth Coeff:{:.6f}".format(float(stealth_coeff)))
        logger.info("Total Loss:{:.6f}".format(loss.item()))
        logger.info("----------------------------------------------------------------")

        opt_gradient = [gamma * grad + belta * ref_grad for grad,
                    ref_grad in zip(gradient_poi_unit, ref_benign_gradient)]
        if unit_dir_opt:
            opt_gradient = copy_gradient_unit(opt_gradient)
            norm_list = cal_layer_gradient_norm(ref_benign_gradient, norm=2)
            opt_gradient = [grad*norm for grad, norm in zip(opt_gradient, norm_list)]

        return _replicate_malicious_gradient(opt_gradient, malicious_num)


def fedSGD_epoch_model_replacement_against_defence(
    server,
    clients,
    train_loaders,
    replacement_state_dict,
    inference_gradient_AGR,
    target_class,
    poi_index,
    logger,
    scale=100,
    coeff=[1, 0, 1, 0],
    optimization_round=100,
    ifprint=False,
    **kwargs
):
    """
    Drop-in AGR-aware stealth replacement poisoning epoch.

    Difference from Main_framework_lib2.fedSGD_epoch_model_replacement_against_defence:
    one shared malicious update is optimized through the weighted mean of all
    client gradients with a stealth penalty, then copied to every malicious
    client.
    """

    loader_size = len(train_loaders[0])
    dataset_size = len(train_loaders[0].dataset)
    client_num = len(clients)
    mal_client_idx = _ordered_malicious_indices(poi_index)

    begin = time.time()
    total_acc, total_loss = 0.0, 0.0
    train_iteration_loaders = [iter(dataloader) for dataloader in train_loaders]

    eval_list = []

    for client in clients:
        client.set_server(server)

    for batch_index in range(0, loader_size):
        gradient_list = [None for _ in range(client_num)]
        bn_data_list = [None for _ in range(client_num)]
        clients_acc, clients_loss = 0.0, 0.0

        benign_gradient_by_idx = {}
        benign_gradient_list = []
        ref_benign_gradient_list = []
        batch_data = []

        for client, train_loader in zip(clients, train_iteration_loaders):
            X, y = next(train_loader)
            batch_data.append((X, y))

        for index, (client, data_pair) in enumerate(zip(clients, batch_data)):
            X, y = data_pair
            if poi_index[index] == 0:
                gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(X, y)
                gradient = _clone_detached_gradient(gradient)
                gradient_list[index] = gradient
                bn_data_list[index] = bn_data
                benign_gradient_by_idx[index] = gradient
                benign_gradient_list.append(gradient)
            else:
                client.client_model.eval()
                ref_benign_gradient = client.cal_gradient(X, y)
                ref_benign_gradient = _clone_detached_gradient(ref_benign_gradient)
                ref_benign_gradient_list.append(ref_benign_gradient)
                bn_data_list[index] = client.get_bn()
                accuracy, loss_value = client.test_batch(X, y)

            clients_acc += accuracy
            clients_loss += loss_value

        if batch_index == 0:
            gradient_AGR = copy.deepcopy(inference_gradient_AGR)
        else:
            current_global_weights = server.get_weight()
            gradient_AGR = infer_global_aggregated_gradient(
                last_global_weights, current_global_weights, scale=scale
            )

        first_malicious_client = clients[mal_client_idx[0]]
        gradient_poi = calculate_replacement_gradient(
            first_malicious_client.client_model, replacement_state_dict, scale=scale
        )
        gradient_poi = _clone_detached_gradient(gradient_poi)

        surrogate_benign_gradient_list, surrogate_benign_weights = (
            _build_surrogate_benign_gradients(
                client_num, poi_index, ref_benign_gradient_list
            )
        )

        malicious_gradients = optimize_mal_by_agr_aware(
            server=server,
            benign_gradient_list=surrogate_benign_gradient_list,
            benign_weights=surrogate_benign_weights,
            gradient_AGR=gradient_AGR,
            gradient_poi=gradient_poi,
            ref_benign_gradient_list=ref_benign_gradient_list,
            poi_index=poi_index,
            logger=logger,
            optimization_round=optimization_round,
            coeff=coeff,
            ifprint=ifprint,
            **kwargs
        )

        for position, client_idx in enumerate(mal_client_idx):
            gradient_list[client_idx] = malicious_gradients[position]

        if batch_index % 100 == 0:
            mean_aggregation = _mean_all_client_gradients(gradient_list)
            agr_output_loss = gradient_distance_loss(mean_aggregation, gradient_AGR)
            logger.info("---------------- Batch %d AGR-aware Stealth Update ----------------" % batch_index)
            logger.info("Mean aggregate distance loss: {:.6f}".format(agr_output_loss.item()))
            if ifprint:
                print("---------------- Batch %d AGR-aware Stealth Update ----------------" % batch_index)
                print("Mean aggregate distance loss:", agr_output_loss.item())

        if batch_index == loader_size - 1:
            eval_list.append(gradient_AGR)
            eval_list.append(gradient_poi)
            eval_list.append(malicious_gradients)

        total_acc += clients_acc / client_num
        total_loss += clients_loss / client_num

        last_global_weights = server.get_weight()
        server.train_sgd_batch(gradient_list, bn_data_list)
        global_model_state_dict = copy.deepcopy(server.despatch())

        for client in clients:
            client.update_local_model(global_model_state_dict)

    if len(eval_list) == 3:
        gradient_AGR = eval_list[0]
        gradient_poi = eval_list[1]
        malicious_gradients = eval_list[2]
        final_full_gradient_list = _build_full_gradient_list(
            client_num,
            mal_client_idx,
            {idx: grad for idx, grad in enumerate(gradient_list) if poi_index[idx] == 0},
            malicious_gradients,
        )
        final_aggregation = _mean_all_client_gradients(final_full_gradient_list)
        agr_output_loss = gradient_distance_loss(final_aggregation, gradient_AGR)
        mal_output_loss = gradient_distance_loss(final_aggregation, gradient_poi)
        print("---------------- Final AGR-aware Stealth Malicious Update ----------------")
        print("Mean aggregate distance loss:", agr_output_loss.item())
        print("Poisoning gradient output distance loss:", mal_output_loss.item())

    train_acc = total_acc / dataset_size
    train_loss = total_loss / dataset_size

    end = time.time()
    time_elapsed = end - begin

    return train_acc, train_loss, time_elapsed
