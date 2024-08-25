import pandas as _pd
from torch.utils.data import Dataset as _Dataset
import pkg_resources as _pkg_resources

class _MyDataset(_Dataset):
    def __init__(self, dataframe, input_columns=None, target_columns=None):
        self.dataframe = dataframe
        # one hot encode 'shims', 'shape', 'material_coil', 'material_yoke', 'symmetry', 'subset'
        self.dataframe = _pd.get_dummies(self.dataframe, columns=['shims', 'shape', 'material_coil', 'material_yoke', 'symmetry', 'coolingRequirementMax', 'subset'])
        # available keys
        # 'name', 'gfr_x', 'gfr_y', 'gfr_margin',
        # 'maxCurrentDensity', 'fieldTolerance', 'aper_x', 'aper_y',
        # 'aper_x_poleoverhang', 'aper_y_distFromCoil', 'aper_x_tapering',
        # 'aper_x_taperingstop', 'B_design', 'B_design_margin', 'B_real',
        # 'coil_width', 'coil_height', 'yoke_x', 'yoke_y', 'maxBmaterial',
        # 'shims', 'shape', 'fillfactor', 'windings', 'symmetry', 'w', 'w_leg',
        # 'totalCurrent', 'totalCurrentMax', 'coilAreaTotal', 'coilWeightTotal',
        # 'coilVolumeTotal', 'coolingRequirementMax', 'length', 'material_coil',
        # 'material_yoke', 'yokeAreaTotal', 'yokeWeightTotal', 'yokeVolumeTotal',
        # 'B0', 'gfr_x_1e-7', 'gfr_x_2e-7', 'gfr_x_5e-7', 'gfr_x_1e-6',
        # 'gfr_x_2e-6', 'gfr_x_5e-6', 'gfr_x_1e-5', 'gfr_x_2e-5', 'gfr_x_5e-5',
        # 'gfr_x_1e-4', 'gfr_x_2e-4', 'gfr_x_5e-4', 'gfr_x_1e-3', 'gfr_x_2e-3',
        # 'gfr_x_5e-3', 'gfr_x_1e-2', 'gfr_y_1e-7', 'gfr_y_2e-7', 'gfr_y_5e-7',
        # 'gfr_y_1e-6', 'gfr_y_2e-6', 'gfr_y_5e-6', 'gfr_y_1e-5', 'gfr_y_2e-5',
        # 'gfr_y_5e-5', 'gfr_y_1e-4', 'gfr_y_2e-4', 'gfr_y_5e-4', 'gfr_y_1e-3',
        # 'gfr_y_2e-3', 'gfr_y_5e-3', 'gfr_y_1e-2'
        # self.input_columns = ['aper_x', 'aper_y', 'fieldTolerance', 'B_design', 'B_real', 'maxCurrentDensity',
        #                         'gfr_x', 'gfr_y', 'gfr_margin', 'aper_x_poleoverhang', 'aper_y_distFromCoil',
        #                         'aper_x_tapering', 'aper_x_taperingstop', 'B_design_margin',
        #                         'coil_width', 'coil_height', 'yoke_x', 'yoke_y', 'maxBmaterial',
        #                         'shims', 'shape', 'fillfactor', 'windings', 'symmetry', 'w', 'w_leg',
        #                         'totalCurrent', 'totalCurrentMax', 'coilAreaTotal', 'coilWeightTotal',
        #                         'coilVolumeTotal', 'coolingRequirementMax', 'length', 'material_coil',
        #                         'material_yoke', 'yokeAreaTotal', 'yokeWeightTotal', 'yokeVolumeTotal']
        self.target_columns = target_columns
        self.input_columns = [col for col in self.dataframe.columns if col not in self.target_columns and col != 'name']
        if input_columns is not None:
            self.input_columns = input_columns
        if target_columns is not None:
            self.target_columns = target_columns
        
        self.inputs = self.dataframe[self.input_columns]
        self.targets = self.dataframe[self.target_columns]

        self.input_mins = self.inputs.min()
        self.input_maxs = self.inputs.max()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        inputs = self.inputs.iloc[idx].values.astype(float)
        target = self.targets.iloc[idx].values.astype(float)
        return inputs, target



# _df_a = _pd.read_csv(_pkg_resources.resource_filename(__name__, './data/hdipole_gfr_dataset_00k-30k.csv'))
# _df_b = _pd.read_csv(_pkg_resources.resource_filename(__name__, './data/hdipole_gfr_dataset_30k-60k.csv'))
_df_a = _pd.read_csv(_pkg_resources.resource_filename(__name__,"md_dipole_hshaped_v2_straight.csv"))
_df_b = _pd.read_csv(_pkg_resources.resource_filename(__name__,"md_dipole_hshaped_v2_random.csv"))
_df_c = _pd.read_csv(_pkg_resources.resource_filename(__name__,"md_dipole_hshaped_v2_random_small.csv"))
_df_d = _pd.read_csv(_pkg_resources.resource_filename(__name__,"md_dipole_hshaped_v2_random_large.csv"))
_df = _pd.concat([_df_a, _df_b, _df_c, _df_d], ignore_index=True)
# exchange all NaN values of Cooling Requirement Max with 'None' (string was misinterpreted as NaN)
_df["coolingRequirementMax"] = _df["coolingRequirementMax"].fillna("None")
_df = _df.drop(columns=["Unnamed: 0"])
# _target_columns = ['B0', 'gfr_x_1e-4', 'gfr_y_1e-4']
_target_columns_dipole_h = ['B0', 'grad0', 'gfr_x_1e-7', 'gfr_x_2e-7', 'gfr_x_5e-7', 'gfr_x_1e-6',
    'gfr_x_2e-6', 'gfr_x_5e-6', 'gfr_x_1e-5', 'gfr_x_2e-5', 'gfr_x_5e-5',
    'gfr_x_1e-4', 'gfr_x_2e-4', 'gfr_x_5e-4', 'gfr_x_1e-3', 'gfr_x_2e-3',
    'gfr_x_5e-3', 'gfr_x_1e-2', 'gfr_y_1e-7', 'gfr_y_2e-7', 'gfr_y_5e-7',
    'gfr_y_1e-6', 'gfr_y_2e-6', 'gfr_y_5e-6', 'gfr_y_1e-5', 'gfr_y_2e-5',
    'gfr_y_5e-5', 'gfr_y_1e-4', 'gfr_y_2e-4', 'gfr_y_5e-4', 'gfr_y_1e-3',
    'gfr_y_2e-3', 'gfr_y_5e-3', 'gfr_y_1e-2']

Dipole_H_unshuffeled = _MyDataset(_df,target_columns=_target_columns_dipole_h)

_df = _df.sample(frac=1, random_state=42)
Dipole_H_train = _MyDataset(_df.iloc[:int(0.8*len(_df))],target_columns=_target_columns_dipole_h)
Dipole_H_val = _MyDataset(_df.iloc[int(0.8*len(_df)):],target_columns=_target_columns_dipole_h)