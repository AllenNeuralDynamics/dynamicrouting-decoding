from __future__ import annotations

# stdlib imports --------------------------------------------------- #
import argparse
import dataclasses
import gc
import json
import functools
import logging
import pathlib
import time
import uuid
from typing import Any, Literal

# 3rd-party imports necessary for processing ----------------------- #
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pynwb
import upath
import zarr
from dynamic_routing_analysis import spike_utils, decoding_utils, data_utils, path_utils


# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(name)s.%(funcName)s | %(message)s",     datefmt="%Y-%d-%m %H:%M:%S",
    )
logger = logging.getLogger(__name__)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


# utility functions ------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=str, default=None)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--override_params_json', type=str, default="{}")
    for field in dataclasses.fields(Params):
        if field.name in [getattr(action, 'dest') for action in parser._actions]:
            # already added field above
            continue
        logger.debug(f"adding argparse argument {field}")  
        if isinstance(field.type, str):
            type_ = eval(field.type)
        else:
            type_ = field.type
        parser.add_argument(f'--{field.name}', type=type_)
    args = parser.parse_known_args()[0]
    logger.info(f"{args=}")
    return args

@functools.cache
def get_datacube_dir() -> pathlib.Path:
    for p in get_data_root().iterdir():
        if p.is_dir() and p.name.startswith('dynamicrouting_datacube'):
            path = p
            break
    else:
        for p in get_data_root().iterdir():
            if any(pattern in p.name for pattern in ('session_table', 'nwb', 'consolidated', )):
                path = get_data_root()
                break
        else:
            raise FileNotFoundError(f"Cannot determine datacube dir: {list(get_data_root().iterdir())=}")
    logger.info(f"Using files in {path}")
    return path

@functools.cache
def get_data_root(as_str: bool = False) -> pathlib.Path:
    expected_paths = ('/data', '/tmp/data', )
    for p in expected_paths:
        if (data_root := pathlib.Path(p)).exists():
            logger.info(f"Using {data_root=}")
        return data_root.as_posix() if as_str else data_root
    else:
        raise FileNotFoundError(f"data dir not present at any of {expected_paths=}")

@functools.cache
def get_nwb_paths() -> tuple[pathlib.Path, ...]:
    return tuple(get_data_root().rglob('*.nwb'))
    

def get_nwb(session_id_or_path: str | pathlib.Path, raise_on_missing: bool = True, raise_on_bad_file: bool = True) -> pynwb.NWBFile:
    if isinstance(session_id_or_path, (pathlib.Path, upath.UPath)):
        nwb_path = session_id_or_path
    else:
        if not isinstance(session_id_or_path, str):
            raise TypeError(f"Input should be a session ID (str) or path to an NWB file (str/Path), got: {session_id_or_path!r}")
        if pathlib.Path(session_id_or_path).exists():
            nwb_path = session_id_or_path
        elif session_id_or_path.endswith(".nwb") and any(p.name == session_id_or_path for p in get_nwb_paths()):
            nwb_path = next(p for p in get_nwb_paths() if p.name == session_id_or_path)
        else:
            try:
                nwb_path = next(p for p in get_nwb_paths() if p.stem == session_id_or_path)
            except StopIteration:
                msg = f"Could not find NWB file for {session_id_or_path!r}"
                if not raise_on_missing:
                    logger.error(msg)
                    return
                else:
                    raise FileNotFoundError(f"{msg}. Available files: {[p.name for p in get_nwb_paths()]}") from None
    logger.info(f"Reading {nwb_path}")
    try:
        nwb = pynwb.NWBHDF5IO(nwb_path).read()
    except RecursionError:
        msg = f"{nwb_path.name} cannot be read due to RecursionError (hdf5 may still be accessible)"
        if not raise_on_bad_file:
            logger.error(msg)
            return
        else:
            raise RecursionError(msg)
    else:
        return nwb

def ensure_nonempty_results_dir() -> None:
    # pipeline can crash if a results folder is expected and not found, and requires creating manually:
    results = pathlib.Path("/results")
    results.mkdir(exist_ok=True)
    if not list(results.iterdir()):
        (results / uuid.uuid4().hex).touch()

# processing function ---------------------------------------------- #
# modify the body of this function, but keep the same signature

def process_session(session_id: str, params: "Params", test: int = 0, skip_existing: bool = False) -> None:
    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    # Get nwb file
    # Currently this can fail for two reasons: 
    # - the file is missing from the datacube, or we have the path to the datacube wrong (raises a FileNotFoundError)
    # - the file is corrupted due to a bad write (raises a RecursionError)
    # Choose how to handle these as appropriate for your capsule

    try:
        session = get_nwb(session_id, raise_on_missing=True, raise_on_bad_file=True) 
    except (FileNotFoundError, RecursionError) as exc:
        logger.info(f"Skipping {session_id}: {exc!r}")
        return
    
    if test:
        params.folder_name = f"test/{params.folder_name}"
        params.only_use_all_units = True
        params.n_units = ["all"]
        params.keep_n_SVDs = 5
        params.LP_parts_to_keep = ["ear_base_l"]
        params.n_repeats = 1
        params.n_unit_threshold = 5
        logger.info(f"Test mode: using modified set of parameters")

    if skip_existing and params.file_path.exists():
        logger.info(f"{params.file_path} exists: processing skipped")
        return
    
    # Get components from the nwb file:
    trials = session.trials[:]
    units = session.units[:].query(params.units_query)
    if test:
        logger.info(f"Test mode: using reduced set of units")
        units = units.sort_values('location').head(20)

    #units: pd.DataFrame =  utils.remove_pynwb_containers_from_table(units[:])
    units['session_id'] = session_id
    units.drop(columns=['waveform_sd','waveform_mean'], inplace=True, errors='ignore')
                                   
    logger.info(f'starting decode_context_with_linear_shift for {session_id} with {params.to_json()}')
    decoding_utils.decode_context_with_linear_shift(session=session,params=params.to_dict(),trials=trials,units=units)

    del units
    del trials
    del session
    gc.collect()

    logger.info(f'making summary tables of decoding results for {session_id}')
    decoding_results = decoding_utils.concat_decoder_results(
        files=[params.file_path],
        savepath=params.savepath,
        return_table=True,
        single_session=True,
    )
    #find n_units to loop through for next step
    n_units = []
    for col in decoding_results.filter(like='true_accuracy_').columns.values:
        if len(col.split('_'))==3:
            temp_n_units=col.split('_')[2]
            try:
                n_units.append(int(temp_n_units))
            except:
                n_units.append(temp_n_units)
        else:
            n_units.append(None)

    decoding_results = []
    for nu in n_units:
        decoding_utils.concat_trialwise_decoder_results(
            files=[params.file_path],
            savepath=params.savepath,
            return_table=False,
            n_units=nu,
            single_session=True,
        )

    logger.info('writing params file for {session_id}')
    params.write_json(params.file_path.with_suffix('.json'))
    
    
# define run params here ------------------------------------------- #

# The `Params` class is used to store parameters for the run, for passing to the processing function.
# @property methods (like `savepath` below) are computed from other parameters on-demand as required:
# this way, we can separate the parameters dumped to json from larger arrays etc. required for
# processing. We can also update other fields during test mode, and the updated values will be incoporated 
# into these fields.

# - if needed, we can get parameters from the command line and pass them to the dataclass (see `main()` below):
#   just add a field to the App Builder parameters with the same `Parameter Name`

# this is an example from Sam's processing code, replace with your own parameters as needed:
@dataclasses.dataclass
class Params:
    # ----------------------------------------------------------------------------------
    # defaults don't matter for these parameters, they will be updated later:
    session_id: str = ""
    run_id: str = ""
    """A unique string that should be attached to all decoding runs in the same batch"""
    # ----------------------------------------------------------------------------------

    folder_name: str = "test"
    unit_criteria: str = 'medium'
    n_units: list = dataclasses.field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 'all'])
    """number of units to sample for each area"""
    n_repeats: int = 25
    """number of times to repeat decoding with different randomly sampled units"""
    input_data_type: str | Literal['spikes', 'facemap', 'LP'] = 'spikes'
    vid_angle_facemotion: str | Literal['behavior', 'face', 'eye'] = 'face'
    vid_angle_LP: str | Literal['behavior', 'face', 'eye'] = 'behavior'
    central_section: str = '4_blocks_plus'
    """or linear shift decoding, how many trials to use for the shift. '4_blocks_plus' is best"""
    exclude_cue_trials: bool = False
    """option to totally exclude autorewarded trials"""
    n_unit_threshold: int = 5
    """minimum number of units to include an area in the analysis"""
    keep_n_SVDs: int = 500
    """number of SVD components to keep for facemap data"""
    LP_parts_to_keep: list = dataclasses.field(default_factory=lambda: ['ear_base_l', 'eye_bottom_l', 'jaw', 'nose_tip', 'whisker_pad_l_side'])
    spikes_binsize: float = 0.2
    spikes_time_before: float = 0.2
    spikes_time_after: float = 0.01
    use_structure_probe: bool = True
    """if True, append probe name to area name when multiple probes in the same area"""
    crossval: Literal['5_fold', 'blockwise'] = '5_fold'
    """blockwise untested with linear shift"""
    labels_as_index: bool = True
    """convert labels (context names) to index [0,1]"""
    decoder_type: str | Literal['linearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'
    only_use_all_units: bool = False
    """if True, do not run decoding with different areas, only with all areas -- for debugging"""

    @property
    def savepath(self) -> upath.UPath:
        return path_utils.DECODING_ROOT_PATH / f"{self.folder_name}_{self.run_id}" 

    @property
    def filename(self) -> str:
        return f"{self.session_id}_{self.run_id}.pkl"

    @property
    def file_path(self) -> upath.UPath:
        return self.savepath / self.filename
    
    @property
    def units_query(self) -> str:
        if self.unit_criteria == 'medium':
            return 'isi_violations_ratio<=0.5 and presence_ratio>=0.9 and amplitude_cutoff<=0.1'
        elif self.unit_criteria == 'strict':
            return 'isi_violations_ratio<=0.1 and presence_ratio>=0.99 and amplitude_cutoff<=0.1'
        else:
            raise ValueError(f"No units query available for {self.unit_criteria=!r}")

    def to_json(self, **dumps_kwargs) -> str:
        """json string of field name: value pairs, excluding values from property getters (which may be large)"""
        return json.dumps(dataclasses.asdict(self), **dumps_kwargs)

    def write_json(self, path: str | upath.UPath = '/results/params.json') -> None:
        path = upath.UPath(path)
        logger.info(f"Writing params to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))

    def to_dict(self) -> dict[str, Any]:
        """dict of field name: value pairs, including values from property getters"""
        return dataclasses.asdict(self) | {k: getattr(self, k) for k in dir(self.__class__) if isinstance(getattr(self.__class__, k), property)}

# ------------------------------------------------------------------ #


def main():
    t0 = time.time()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:
    params = {}
    for field in dataclasses.fields(Params):
        if (val := getattr(args, field.name, None)) is not None:
            params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            params[k] = v
    
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:    
    session_table = pd.read_parquet(get_datacube_dir() / 'session_table.parquet')
    session_ids: list[str] = session_table.query(
        "is_ephys & project=='DynamicRouting' & is_task & is_annotated & ~is_context_naive"
    )['session_id'].values.tolist()
    if args.session_id is not None:
        if args.session_id not in session_ids:
            logger.info(f"{args.session_id!r} not in filtered sessions list")
            exit()
        logger.info(f"Using single session_id {args.session_id} provided via command line argument")
        session_ids = [args.session_id]
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids")

    # run processing function for each session, with test mode implemented:
    for session_id in session_ids:
        process_session(session_id, params=Params(**params | {'session_id': session_id}), test=args.test, skip_existing=args.skip_existing)
        if args.test:
            logger.info("Test mode: exiting after first session")
            break
    ensure_nonempty_results_dir()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
