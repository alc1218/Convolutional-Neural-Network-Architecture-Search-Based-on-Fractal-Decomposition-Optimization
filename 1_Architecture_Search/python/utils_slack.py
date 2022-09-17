from urllib import request, parse
import json
import constants
import numpy as np

from prettytable import PrettyTable


def send_message_to_slack(text, hook_url):
    post = {"text": "{0}".format(text)}
    try:
        json_data = json.dumps(post)
        req = request.Request(hook_url,
                              data=json_data.encode('ascii'),
                              headers={'Content-Type': 'application/json'})
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))


def top_N_to_table(top_N):
    table = PrettyTable()
    table.field_names = ["Rank","Arch id", "Best val acc", "Architecture"]
    max_acc_list = [round(np.max(val_acc_history)*100, 4) for val_acc_history in top_N['top_N_val_acc']]
    sort_index = np.argsort(max_acc_list)

    for rank, idx in enumerate(sort_index[::-1]):
        arch_id = top_N['top_N_arch_id'][idx]
        max_acc = max_acc_list[idx]
        solution_archi = top_N['top_N_solution_arch'][idx]
        if idx != len(top_N['top_N_val_acc']) - 1:
            row = [str(rank), str(arch_id), str(max_acc), str(solution_archi)]
        else:
            row = ['*'+str(rank + 1)+'*','*'+str(arch_id)+'*', '*'+str(max_acc)+'*', '*'+str(solution_archi)+'*']
        table.add_row(row)
    table_str = table.get_string()
    title = "Search of {}".format(constants.readable_datetime) + '\n'
    return title + table_str


def send_top_N_to_slack(top_N):
    if constants.slack_url is not None:
        send_message_to_slack(top_N_to_table(top_N), constants.slack_url)
