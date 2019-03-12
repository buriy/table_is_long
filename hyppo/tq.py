import tqdm


# noinspection PyBroadException
def is_notebook():
    try:
        try:
            from ipywidgets import FloatProgress
        except ImportError:
            from IPython.html.widgets.widget_float import FloatProgress

        from IPython import get_ipython
        ipython = get_ipython()
        if not ipython or ipython.__class__.__name__ != 'ZMQInteractiveShell':
            return False
    except Exception:
        return False
    return True


if is_notebook():
    tq = tqdm.tqdm_notebook
else:
    tq = tqdm.tqdm
