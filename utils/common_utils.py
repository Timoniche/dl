import datetime
import pathlib


def cur_dir():
    return str(pathlib.Path().resolve())


def parent_dir():
    return str(pathlib.Path().resolve().parent)


def generate_time_id():
    current_datetime = datetime.datetime.now()
    time_id = current_datetime.strftime('%Y_%m_%d_%H_%M')

    return time_id
