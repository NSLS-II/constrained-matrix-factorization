from pathlib import Path
import numpy as np
from time import sleep, time
import matplotlib.pyplot as plt
from deepnmf.companion.nmf import decomposition, iterative_decomposition, example_plot
from IPython import display


# TODO: intial component dir
class DirectoryAgent:
    def __init__(
        self,
        data_dir,
        n_components,
        *,
        data_spec=None,
        x_lim=None,
        component_dir=None,
        output_dir=None,
        header=0,
        file_ordering=None,
        file_limit=None,
        figsize=None,
        **kwargs,
    ):
        """
        Class for building trained model and classifying a directory.
        Classification can be accomplished on the fly or once using the spin() method.

        Parameters
        ----------
        data_dir: pathlike
            Directory containing data to be classified
        n_components:  int
            Number of components for NMF
        data_spec: basestring
            String specification for glob() method in searching data_dir
            Default behavior is to include all files. If writing temporary files, it is important to include
            final file spec such as '*.xy'.
        x_lim:  tuple
            Size two tuple for bounding the Xs to a region of interest
        component_dir: pathlike
            Directory containing initial components.
        header: int
            Number of header lines in file
        training_output_dir: pathlike
            Output directory of training containing checkpoints for loading
        path_to_model: pathlike
            path to model to load in full (not presently implemented)
        file_ordering: function
            Function for sorting file paths as key argument in sorted
        file_limit: int
            Maximum number of files to consider
        figsize: tuple
            Two integer tuple for matplotlib figsize. Keep in mind all plots appear in a row.
        **kwargs:
            Keyword arguments to pass to companion.nmf.decomposition
        """

        self.dir = Path(data_dir).expanduser()
        if component_dir is None:
            self.component_dir = None
        else:
            self.component_dir = Path(component_dir).expanduser()
        self.n_components = n_components
        if data_spec is None:
            self.path_spec = "*"
        else:
            self.path_spec = data_spec
        self.output_dir = output_dir
        self.paths = []
        self.Ys = []
        self.Xs = []
        self.initial_components = []
        self.limit = file_limit
        self.x_lim = x_lim
        self.header = header
        self.fig = plt.figure(figsize=figsize)
        self.decomposition_args = kwargs

        # TODO: Elegantly understand if I'm in a GUI or in ipython inline
        try:
            self.fig.canvas.manager.show()
        except:
            display.display(self.fig)

        if file_ordering is None:
            self.file_ordering = lambda x: x
        else:
            self.file_ordering = file_ordering

    def __len__(self):
        return len(self.paths)

    def path_list(self):
        return sorted(list(self.dir.glob(self.path_spec)), key=self.file_ordering)

    def load_files(self, paths):
        xs = []
        ys = []
        paths = sorted(paths, key=self.file_ordering)
        for idx, path in enumerate(paths):
            if not (self.limit is None) and idx >= self.limit:
                break
            _x, _y = np.loadtxt(path, comments="#", skiprows=self.header).T
            xs.append(_x)
            ys.append(_y)
        return xs, ys

    def _decomposition(self, *args, **kwargs):
        return decomposition(*args, **kwargs)

    def update_plot(self):
        if len(self) < 2:
            return
        Xs = np.array(self.Xs)
        Ys = np.array(self.Ys)
        sub_X, sub_Y, alphas, components = self._decomposition(
            Xs,
            Ys,
            q_range=self.x_lim,
            n_components=self.n_components,
            normalize=True,
            initial_components=self.initial_components,
            fix_components=[True for _ in range(len(self.initial_components))],
            **self.decomposition_args,
        )

        self.fig.clf()
        axes = self.fig.subplots(1, self.n_components + 2)
        example_plot(
            sub_X,
            sub_Y,
            alphas,
            axes=axes[:-1],
            sax=axes[-2],
            components=components,
            comax=axes[-1],
            summary_fig=True,
        )

        # TODO: Elegantly understand if I'm in a GUI or in ipython inline
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        display.clear_output(wait=True)
        display.display(self.fig)

    def spin(self, sleep_delay=60, verbose=False, timeout=0):
        """
        Starts the spin to read new files and append their classifications to the output and internal dictionary
        If a single pass of the data_directory is required, use a short or negative timeout time.
        This can be run as a multiprocessing.Process target with a Manager to retain the output list if being run dynamically.

        Parameters
        ----------
        sleep_delay: float
            number of seconds to wait before checking directory for new files
        verbose: bool
            Print classifications to screen
        timeout: float
            Time to wait before stop spinning. Default is infinite spin. If a single pass is required, use a negative value.

        Returns
        -------
        self.classifications:
            dictionary of file basenames and classifications

        """
        start_time = time()

        while True:
            if len(self.path_list()) != len(self):
                self.fig.clf()
                for path in self.path_list():
                    if path.name not in self.paths:
                        self.paths.append(path.name)
                        xs, ys = self.load_files([path])
                        self.Xs.extend(xs)
                        self.Ys.extend(ys)
            if self.component_dir:
                _, self.initial_components = self.load_files(
                    list(self.component_dir.glob(self.path_spec))
                )
            if len(self.initial_components) > self.n_components:
                raise RuntimeError(
                    f"Found {len(self.initial_components)} files for initial components, "
                    f"but the NMF is set to use only {self.n_components} components"
                )
            self.update_plot()
            if timeout and time() - start_time > timeout:
                break
            sleep(sleep_delay)
        return np.array(self.Xs), np.array(self.Ys)

    def elbow_plot(self, n_max):
        import torch
        from deepnmf.nmf.utils import sweep_components

        xs, ys = self.load_files(self.path_list())
        Y = torch.tensor(ys)
        return sweep_components(Y, n_max=n_max)


class AutoDirectoryAgent(DirectoryAgent):
    def __init__(
        self,
        data_dir,
        n_components,
        *,
        data_spec=None,
        x_lim=None,
        output_dir=None,
        header=0,
        file_ordering=None,
        file_limit=None,
        figsize=None,
        **kwargs,
    ):
        super().__init__(
            data_dir,
            n_components,
            data_spec=data_spec,
            x_lim=x_lim,
            output_dir=output_dir,
            header=header,
            file_ordering=file_ordering,
            file_limit=file_limit,
            figsize=figsize,
            **kwargs,
        )

    def _decomposition(self, *args, **kwargs):
        return iterative_decomposition(*args, **kwargs)


# TODO: Implement RemoteDispatcher and Accumulator agents. See
#  https://github.com/NSLS-II-XPD/ae-gpcam/blob/main/companion/agent.py
