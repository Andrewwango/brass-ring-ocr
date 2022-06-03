"""
Misc utils for notebook
"""

def read_log(log):
    """Read log generated from DTRB output

    Args:
        log (str): log filename

    Returns:
        list: one image file per line, list of words for each line
    """
    with open(log, "r") as f:
        lines = [line[:-2].split(",") for line in f.readlines()]
    return lines

