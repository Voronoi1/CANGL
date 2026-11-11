#!/usr/bin/env python
# coding: utf-8

# Main - Step 3 alternative: GI-NAS inversion with a preliminary normal
# aggregation followed by an AGR-aware stealth submission.

import copy
import torch

import global_var


device = "cuda" if torch.cuda.is_available() else "cpu"
global_var.set_device(device)

from dataset import get_dataset
from gradient_lib import cal_gradient_distance_array2, cal_gradient_mean
from Main_framework_lib3 import gradient_simulate
from Main_framework_lib2_AGRAware_Stealth import (
    _build_surrogate_benign_gradients,
    _clone_detached_gradient,
    _ordered_malicious_indices,
    optimize_mal_by_agr_aware,
)
from Main_framework_utils import (
    calculate_replacement_gradient,
    infer_global_aggregated_gradient,
)
from ginas_inversion import inversion_ginas


INITIAL_GLOBAL_MODEL_PATH = "./model/Cifar100_FK_0_MKrum_AGRAware_Stealth_Poisoning_Epoch002.pth"
MALICIOUS_MODEL_PATH = "./model/Cifar100_FK_0_Malicious_Model.pth"


def agr_aware_stealth_second_submission(
    server,
    clients,
    input_batches,
    replacement_state_dict,
    previous_aggregated_gradient,
    runtime_logger,
    coeff=None,
    poi_index=None,
    return_candidx=True,
    scale=100,
    **optimization_kwargs
):
    """Keep the second aggregation close to the first aggregated gradient."""
    if coeff is None:
        coeff = [1, 0, 0, 0]
    if poi_index is None:
        raise ValueError("poi_index is required for the second submission.")
    if previous_aggregated_gradient is None:
        raise ValueError(
            "The first aggregated gradient is required for the second submission."
        )

    client_num = len(clients)
    malicious_indices = _ordered_malicious_indices(poi_index)
    if not malicious_indices:
        raise ValueError("At least one malicious client is required.")

    for client in clients:
        client.set_server(server)

    last_global_weights = server.get_weights()
    previous_global_model_state_dict = copy.deepcopy(
        server.global_model.state_dict()
    )

    # Match the FK stealth optimizer used by the poisoning script: each
    # malicious client first computes a clean local reference gradient, and
    # those references represent the unknown benign-client slots by weight.
    ref_benign_gradient_list = []
    for client_index in malicious_indices:
        client = clients[client_index]
        X, y = input_batches[client_index]
        client.client_model.eval()
        ref_benign_gradient_list.append(
            _clone_detached_gradient(client.cal_gradient(X, y))
        )

    surrogate_gradients, surrogate_weights = _build_surrogate_benign_gradients(
        client_num,
        poi_index,
        ref_benign_gradient_list,
    )

    first_malicious_client = clients[malicious_indices[0]]
    replacement_gradient = calculate_replacement_gradient(
        first_malicious_client.client_model,
        replacement_state_dict,
        scale=scale,
    )
    replacement_gradient = _clone_detached_gradient(replacement_gradient)
    first_aggregated_gradient = _clone_detached_gradient(
        previous_aggregated_gradient
    )

    malicious_gradients = optimize_mal_by_agr_aware(
        server=server,
        benign_gradient_list=surrogate_gradients,
        benign_weights=surrogate_weights,
        gradient_AGR=first_aggregated_gradient,
        gradient_poi=replacement_gradient,
        ref_benign_gradient_list=ref_benign_gradient_list,
        poi_index=poi_index,
        logger=runtime_logger,
        coeff=coeff,
        ifprint=False,
        **optimization_kwargs
    )

    malicious_positions = {
        client_index: position
        for position, client_index in enumerate(malicious_indices)
    }
    gradient_list = []
    bn_data_list = []
    for client_index, (client, input_tuple) in enumerate(
        zip(clients, input_batches)
    ):
        X, y = input_tuple
        if client_index in malicious_positions:
            gradient = malicious_gradients[malicious_positions[client_index]]
            accuracy, loss_value = client.test_batch(X, y)
            bn_data = client.get_bn()
        else:
            gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(
                X,
                y,
            )
        gradient_list.append(gradient)
        bn_data_list.append(bn_data)

    _, candidate_indices = server.robust_aggregate_grads(
        gradient_list,
        return_candidx=True,
    )
    server.train_sgd_batch(gradient_list, bn_data_list)
    current_global_weights = server.get_weights()
    inference_gradient = infer_global_aggregated_gradient(
        last_global_weights,
        current_global_weights,
        scale=scale,
    )

    print("#Client:", client_num)
    print("Lr:", server.get_lr())

    result = (
        gradient_list,
        inference_gradient,
        previous_global_model_state_dict,
    )
    if return_candidx:
        return result + (candidate_indices,)
    return result


def build_krum_then_filter_aggregator(client_num, malicious_num):
    target_client_num = 2 * malicious_num + 3
    target_client_indices = range(target_client_num)

    def aggregate(
        gradient_list,
        n_attackers=None,
        mal_client_idx=None,
        return_candidx=False,
    ):
        if len(gradient_list) != client_num:
            raise ValueError(
                "Expected {} client gradients, got {}.".format(
                    client_num,
                    len(gradient_list),
                )
            )

        attacker_num = malicious_num if n_attackers is None else n_attackers
        distance_array = cal_gradient_distance_array2(gradient_list)

        initial_neighbor_num = len(gradient_list) - 2 - attacker_num
        target_scores = {
            client_index: sum(
                sorted(
                    distance_array[client_index][other_index]
                    for other_index in range(len(gradient_list))
                    if other_index != client_index
                )[:initial_neighbor_num]
            )
            for client_index in target_client_indices
        }

        remaining_indices = list(range(len(gradient_list)))
        krum_candidate_indices = []
        while len(remaining_indices) > 2 * attacker_num + 2:
            neighbor_num = len(remaining_indices) - 2 - attacker_num
            krum_scores = {}
            for client_index in remaining_indices:
                neighbor_distances = sorted(
                    distance_array[client_index][other_index]
                    for other_index in remaining_indices
                    if other_index != client_index
                )
                krum_scores[client_index] = sum(
                    neighbor_distances[:neighbor_num]
                )

            selected_index = min(
                remaining_indices,
                key=lambda client_index: krum_scores[client_index],
            )
            krum_candidate_indices.append(selected_index)
            remaining_indices.remove(selected_index)

        selected_target_indices = [
            index
            for index in krum_candidate_indices
            if index < target_client_num
        ]
        if selected_target_indices:
            retained_target_index = min(
                selected_target_indices,
                key=lambda client_index: target_scores[client_index],
            )
            candidate_indices = [
                index
                for index in krum_candidate_indices
                if index >= target_client_num or index == retained_target_index
            ]
        else:
            candidate_indices = krum_candidate_indices

        candidate_indices.sort()
        aggregation = cal_gradient_mean(
            [gradient_list[index] for index in candidate_indices]
        )

        if return_candidx:
            return aggregation, candidate_indices
        return aggregation

    return aggregate


def build_data_info():
    cifar100_train_dataset = get_dataset("Cifar100", train=True, transform=None, download=True)
    cifar100_test_dataset = get_dataset("Cifar100", train=False, transform=None, download=True)

    cifar100_train_loader = torch.utils.data.DataLoader(
        cifar100_train_dataset,
        batch_size=128,
        shuffle=True,
    )
    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test_dataset,
        batch_size=128,
        shuffle=False,
    )

    tiny_train_dataset = get_dataset("TinyImageNet", train=True, transform=None, download=True)
    tiny_test_dataset = get_dataset("TinyImageNet", train=False, transform=None, download=True)

    tiny_train_loader = torch.utils.data.DataLoader(tiny_train_dataset, batch_size=32, shuffle=True)
    tiny_test_loader = torch.utils.data.DataLoader(tiny_test_dataset, batch_size=32, shuffle=False)

    caltech_train_dataset = get_dataset("CalTech256", train=True, transform=None, download=True)
    caltech_test_dataset = get_dataset("CalTech256", train=False, transform=None, download=True)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_train_dataset, batch_size=16, shuffle=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_test_dataset, batch_size=16, shuffle=False)

    return {
        "test_dataset_dict": {
            "TinyImageNet": tiny_test_dataset,
            "Cifar100": cifar100_test_dataset,
            "CalTech256": caltech_test_dataset,
        },
        "train_dataset_dict": {
            "TinyImageNet": tiny_train_dataset,
            "Cifar100": cifar100_train_dataset,
            "CalTech256": caltech_train_dataset,
        },
        "test_loader_dict": {
            "TinyImageNet": tiny_test_loader,
            "Cifar100": cifar100_test_loader,
            "CalTech256": caltech_test_loader,
        },
        "train_loader_dict": {
            "TinyImageNet": tiny_train_loader,
            "Cifar100": cifar100_train_loader,
            "CalTech256": caltech_train_loader,
        },
    }


def run_default_cifar100_mkrum():
    fl_setting = {
        "batch_size": 32,
        "client_num": 50,
        "malicious_num": 10,
    }
    target_class = 0
    dataset_name = "Cifar100"
    knowledge_level = "FK"
    agr_name = "MKrum"
    agr_attack_setting_dict = {
        "optimization_round": 1000,
        "threshold": [8e3, 1e3, 1e4],
        "init_value": [10, 1],
        "lr": 1e-5,
        "opt_parameter": "both",
    }
    sp = False
    robust_aggregator = build_krum_then_filter_aggregator(
        fl_setting["client_num"],
        fl_setting["malicious_num"],
    )

    data_info = build_data_info()
    gradient_tuple = gradient_simulate(
        dataset_name,
        data_info,
        target_class,
        knowledge_level,
        agr_name,
        fl_setting,
        agr_attack_setting_dict,
        sp,
        initial_global_model_path=INITIAL_GLOBAL_MODEL_PATH,
        robust_aggregator=robust_aggregator,
        malicious_model_path=MALICIOUS_MODEL_PATH,
        second_submission_fn=agr_aware_stealth_second_submission,
    )

    ginas_config = {
        "num_iterations": 20000,
        "lr": 1e-3,
        "search_base_channels": 64,
        "train_base_channels": 128,
        "search_random_candidates": 10,
        "max_search_candidates": 18,
        "regularization_mode": "tv_l2",
        "tv_l2_tv_reduction": "variation",
        "l2_coeff": 1e-5,
        "tv_coeff": 1e-4,
        "save_every": 100,
        "seed": 0,
    }
    return inversion_ginas(
        dataset_name,
        target_class,
        knowledge_level,
        agr_name,
        gradient_tuple,
        ginas_config=ginas_config,
        output_size=1,
        ifprint=True,
    )


if __name__ == "__main__":
    run_default_cifar100_mkrum()
