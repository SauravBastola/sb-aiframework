import requests
from pathlib import Path
from typing import List


url = "http://localhost:8080/"

session = requests.Session()


def req(endpoint):
    response = session.get(url + endpoint)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Error:", response.status_code)
        raise Exception(f"Failed endpoint: {url+endpoint} {response.content.decode()}")

    return data


def print_board(board):
    arr2 = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
    c = 0
    for i in range(3):
        for j in range(3):
            if board[c] == 0:
                arr2[i][j] = "O"
            elif board[c] == 1:
                arr2[i][j] = "X"
            else:
                arr2[i][j] = "_"
            c += 1

    for line in arr2:
        print(line)


def copy_network(model):
    # Create a new network with the same architecture as the original
    new_model = type(model)(**model.__dict__)

    # Copy parameters by detaching the graph
    for name, param in model.named_parameters():
        new_model._parameters[name] = param.detach()

    # Optionally, copy buffers if needed (not always necessary)
    for name, buffer in model.named_buffers():
        new_model._buffers[name] = buffer.clone()

    return new_model


def ensure_path_parents(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def index_of_max(logits, mask):
    max_index = None
    max_value = 0
    for action, valid in enumerate(mask):
        if valid:
            logit = logits[action].item()
            if max_index == None or max_value < logit:
                max_index = action
                max_value = logit

    assert max_index != None
    return max_index


def rotate_list(li: List):
    li = [li[-1]] + li[:-1]
    return li
