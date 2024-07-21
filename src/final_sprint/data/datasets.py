import pandas as _pd
from torch.utils.data import Dataset as _Dataset
import pkg_resources as _pkg_resources

class _MyDataset(_Dataset):
    def __init__(self, dataframe, input_columns=None, target_columns=None):
        self.dataframe = dataframe
        # one hot encode 'shims', 'shape', 'material_coil', 'material_yoke', 'symmetry'
        self.dataframe = _pd.get_dummies(self.dataframe, columns=['shims', 'shape', 'material_coil', 'material_yoke', 'symmetry', 'coolingRequirementMax'])
       
        self.target_columns = ['B0', 'gfr_x_1e-7', 'gfr_x_2e-7', 'gfr_x_5e-7', 'gfr_x_1e-6',
                                'gfr_x_2e-6', 'gfr_x_5e-6', 'gfr_x_1e-5', 'gfr_x_2e-5', 'gfr_x_5e-5',
                                'gfr_x_1e-4', 'gfr_x_2e-4', 'gfr_x_5e-4', 'gfr_x_1e-3', 'gfr_x_2e-3',
                                'gfr_x_5e-3', 'gfr_x_1e-2', 'gfr_y_1e-7', 'gfr_y_2e-7', 'gfr_y_5e-7',
                                'gfr_y_1e-6', 'gfr_y_2e-6', 'gfr_y_5e-6', 'gfr_y_1e-5', 'gfr_y_2e-5',
                                'gfr_y_5e-5', 'gfr_y_1e-4', 'gfr_y_2e-4', 'gfr_y_5e-4', 'gfr_y_1e-3',
                                'gfr_y_2e-3', 'gfr_y_5e-3', 'gfr_y_1e-2']
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



_df_a = _pd.read_csv(_pkg_resources.resource_filename(__name__, 'hdipole_gfr_dataset_00k-30k.csv'))
_df_b = _pd.read_csv(_pkg_resources.resource_filename(__name__, 'hdipole_gfr_dataset_30k-60k.csv'))
_df = _pd.concat([_df_a, _df_b], ignore_index=True)
# exchange all NaN values of Cooling Requirement Max with 'None' (string was misinterpreted as NaN)
_df["coolingRequirementMax"] = _df["coolingRequirementMax"].fillna("None")
_df = _df.drop(columns=["Unnamed: 0"])
# _target_columns = ['B0', 'gfr_x_1e-4', 'gfr_y_1e-4']
_target_columns = None

Dipole_H_unshuffeled = _MyDataset(_df, target_columns=_target_columns)

_df = _df.sample(frac=1, random_state=42)
Dipole_H_train = _MyDataset(_df.iloc[:int(0.8*len(_df))], target_columns=_target_columns)
Dipole_H_val = _MyDataset(_df.iloc[int(0.8*len(_df)):], target_columns=_target_columns)