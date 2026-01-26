import torch
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import deepmimo as dm
# %%
class MySeqDataLoader(torch.utils.data.Dataset):

    def __init__(self, scenario, tx_sets="all", seed=42, shuffle=False, pad_value=500, 
                 train=True, split_by="users", train_ratio=0.8, sort_by="power"):
        ### get length of dataset
        self.dataset = scenario
        self.Txs = 1
        self.pad_value = pad_value
        self.split_by = split_by
        self.sort_by = sort_by
        self.dataset_filtered = defaultdict(list)
        self.total_length = 0

     
        # if (type(self.dataset) != type([])) or hasattr(self.dataset, 'los'):
        if isinstance(self.dataset.n_ue, int):
        
            self.dataset = [self.dataset]
        
        if split_by == "Tx":
            self.Txs = list(range(len(self.dataset)))
            if train:
                self.Txs = self.Txs[:int(train_ratio * len(self.Txs))]
            else:

                self.Txs = self.Txs[int(train_ratio * len(self.Txs)):]

            for tx in self.Txs:
                use_indices = self.dataset[tx].los != -1
                n_ue = self.dataset[tx].n_ue
            
                for k in self.dataset[tx]:
                    if k not in ["txrx", "load_params", "name", "rt_params", "materials", "scene", "n_ue"]:
                        if self.dataset[tx][k].shape[0] == n_ue:
                            self.dataset_filtered[k].extend(self.dataset[tx][k][use_indices].tolist())
                self.dataset_filtered["tx_pos"].extend(np.repeat(self.dataset[tx]["tx_pos"], n_ue, axis=0).tolist())

        elif split_by == "user":
            print("User-level splitting", len(self.dataset))
            for tx in range(len(self.dataset)):
                n_ue = self.dataset[tx].n_ue
                print("tx_pos:", n_ue)

                # --- user-level splitting ---
                indices = np.arange(n_ue)
                np.random.seed(seed + tx)
                np.random.shuffle(indices)

                split_idx = int(train_ratio * len(indices))
                if train:
                    indices = indices[:split_idx]
                else:
                    indices = indices[split_idx:]

                use_indices = self.dataset[tx].los != -1
                indices = [i for i in indices if use_indices[i]]

                # Collect data
                for k in self.dataset[tx]:
                    if k in ["txrx", "load_params", "name", "rt_params", "materials", "scene", "n_ue"]:
                        continue
                    if self.dataset[tx][k].shape[0] == n_ue:
                        self.dataset_filtered[k].extend(self.dataset[tx][k][indices].tolist())

                # TX position duplicated for each UE sample
                if np.array(self.dataset[tx]["tx_pos"]).ndim == 1:
                    self.dataset_filtered["tx_pos"].extend(
                        np.repeat(self.dataset[tx]["tx_pos"][np.newaxis, :], len(indices), axis=0).tolist()
                    )
                else:
                    self.dataset_filtered["tx_pos"].extend(
                        np.repeat(self.dataset[tx]["tx_pos"], len(indices), axis=0).tolist()
                    )

        
        self.seed = seed
        self.total_length = len(self.dataset_filtered[list(self.dataset_filtered.keys())[0]])

        # boundary = self.dataset[0]['rt_params']['raw_params']['studyarea']['boundary']['data']
        # self.mins = np.array([boundary[0][0], boundary[0][1], 
        #                      self.dataset[0]['rt_params']['raw_params']['studyarea']['boundary']['values']['zmin']]).astype(np.float32)
        # self.maxs = np.array([boundary[2][0], boundary[2][1], 
        #                      self.dataset[0]['rt_params']['raw_params']['studyarea']['boundary']['values']['zmax']]).astype(np.float32)

    def decode_interaction_to_multilabel(self, inter_code):
        """
        Convert interaction code to multi-label vector [R, D, S, T]
        Ignores repetitions: RRD -> RD, RR -> R
        
        Returns:
            np.array of shape (4,): binary indicators for [R, D, S, T]
            Returns [-1, -1, -1, -1] for invalid/NaN codes
        """
        if np.isnan(inter_code):
            return np.array([-1, -1, -1, -1], dtype=np.float32)
        
        code_str = str(int(inter_code))
        multi_label = np.zeros(4, dtype=np.float32)
        
        # Map: 1->R, 2->D, 3->S, 4->T
        for digit in code_str:
            if digit == '1':
                multi_label[0] = 1  # R
            elif digit == '2':
                multi_label[1] = 1  # D
            elif digit == '3':
                multi_label[2] = 1  # S
            elif digit == '4':
                multi_label[3] = 1  # T
        
        return multi_label

    def __getitem__(self, idx):
        prompt = []
        for k in ["tx_pos", "rx_pos"]:
 
            prompt.extend(self.dataset_filtered[k][idx])
        
        # Sort paths based on sort_by option
        if self.sort_by == "power":
            indices = np.argsort(-np.array(self.dataset_filtered["power"][idx]))
        elif self.sort_by == "delay":
            indices = np.argsort(np.array(self.dataset_filtered["delay"][idx]))
        else:
            raise ValueError(f"Unknown sort_by option: {self.sort_by}")

        paths = []
        interactions = []  # NEW: multi-label interactions
        valid_paths = 0
        
        # SOS token
        paths.append([0.0, 0.0, 0.0, 0.0, 0.0])
        interactions.append([-1, -1, -1, -1])  # SOS has no interaction label
        
        for step_idx, indx in enumerate(indices):
            output_per_step = []
            broken = False
            
            # now also include aoa azimuth and elevation
            for k in ["delay", "power", "phase", "aoa_az", "aoa_el"]:
                value = self.dataset_filtered[k][idx][indx]
                if np.isnan(value):
                    value = self.pad_value
                    broken = True
                    break
                elif k == "delay":
                    value = value * 1e6  # convert to us
                elif k == "phase":
                    value = value * (np.pi/180)
                elif k in ["aoa_az", "aoa_el"]:
                    # angles in degrees -> convert to radians
                    value = value * (np.pi/180)
                elif k == "power":
                    value = value * 0.01
                
                output_per_step.append(value)
            
            if not broken:
                valid_paths += 1
                paths.append(output_per_step)
                
                # NEW: Extract and decode interaction
                inter_value = self.dataset_filtered["inter"][idx][indx]
                inter_label = self.decode_interaction_to_multilabel(inter_value)
                interactions.append(inter_label)
        
        num_paths = [valid_paths]
        
        return (torch.tensor(prompt, dtype=torch.float32), 
            torch.tensor(paths, dtype=torch.float32), 
            torch.tensor(num_paths, dtype=torch.float32) / 25.0,
            torch.tensor(interactions, dtype=torch.float32))  # NEW

    def __len__(self):
        return self.total_length

    def collate_fn(self, batch):
        batch_prompts = torch.cat([i[0].unsqueeze(0) for i in batch], dim=0)
        batch_paths = [i[1] for i in batch]
        batch_paths = torch.nn.utils.rnn.pad_sequence(batch_paths, batch_first=True, 
                                                       padding_value=self.pad_value)
        batch_num_paths = [i[2] for i in batch]
        batch_num_paths = torch.nn.utils.rnn.pad_sequence(batch_num_paths, batch_first=True, 
                                                           padding_value=0)
        
        # NEW: collate interactions
        batch_interactions = [i[3] for i in batch]
        batch_interactions = torch.nn.utils.rnn.pad_sequence(batch_interactions, batch_first=True,
                                                             padding_value=-1)
        
        return batch_prompts, batch_paths, batch_num_paths, batch_interactions



class PreTrainMySeqDataLoader(torch.utils.data.Dataset):

    def __init__(self, scenario, tx_sets="all", seed=42, shuffle=False, pad_value=500,
                 train=True, split_by="users", train_ratio=0.8, sort_by="power", normalizers = None, apply_normalizers= []):
        ### get length of dataset
        self.dataset = scenario
        self.Txs = 1
        self.pad_value = pad_value
        self.split_by = split_by
        self.sort_by = sort_by
        self.dataset_filtered = defaultdict(list)
        self.total_length = 0
        scattering_model = { None:0, 'none':0, 'directive':1, }
        self.normalizers = normalizers
        self.apply_normalizers = apply_normalizers
        if isinstance(self.dataset.n_ue, int):
        
            self.dataset = [self.dataset]

        if split_by == "Tx":
            self.Txs = list(range(len(self.dataset)))
            if train:
                self.Txs = self.Txs[:int(train_ratio * len(self.Txs))]
            else:
                self.Txs = self.Txs[int(train_ratio * len(self.Txs)):]

            for tx in self.Txs:
                use_indices = self.dataset[tx].los != -1
                n_ue = self.dataset[tx].n_ue

                for k in self.dataset[tx]:
                    if k not in ["txrx", "load_params", "name", "rt_params", "materials", "scene", "n_ue"]:
                        if self.dataset[tx][k].shape[0] == n_ue:
                            self.dataset_filtered[k].extend(self.dataset[tx][k][use_indices].tolist())
                self.dataset_filtered["tx_pos"].extend(np.repeat(self.dataset[tx]["tx_pos"], n_ue, axis=0).tolist())

        elif split_by == "user":
            for tx in range(len(self.dataset)):
                n_ue = self.dataset[tx].n_ue

                # --- user-level splitting ---
                indices = np.arange(n_ue)
                np.random.seed(seed + tx)
                np.random.shuffle(indices)

                split_idx = int(train_ratio * len(indices))
                if train:
                    indices = indices[:split_idx]
                else:
                    indices = indices[split_idx:]

                use_indices = self.dataset[tx].los != -1
                indices = [i for i in indices if use_indices[i]]

                # Collect data
                for k in self.dataset[tx]:
                    # if k in ["txrx", "load_params", "name", "rt_params", "materials", "scene", "n_ue"]:
                    if k in ["txrx", "load_params", "name", "rt_params","n_ue",]:

                        continue
                    elif k == "scene":
                        self.dataset_filtered["num_materials"].extend( [len(self.dataset[tx][k].objects)]*len(indices))
                        self.dataset_filtered["bounding_box"].extend([abs(self.dataset[tx][k].bounding_box.bounds[1] - self.dataset[tx][k].bounding_box.bounds[0]).tolist()] * len(indices))


                    elif k == "materials":
                        all_material_types = []
                        for material in self.dataset[tx][k]:
                            building_types = []
                            for material_property in ["permittivity", "conductivity", "scattering_coefficient", "scattering_model", "thickness", "roughness"]:
                                if material_property == "scattering_model":

                                    building_types.append(scattering_model[material.__dict__[material_property]] )
                                else:
                                    building_types.append( material.__dict__[material_property] )
                            all_material_types.append(building_types)

                        self.dataset_filtered["material_properties"].extend( [all_material_types]*len(indices))

                    elif self.dataset[tx][k].shape[0] == n_ue:
                        self.dataset_filtered[k].extend(self.dataset[tx][k][indices].tolist())

                # TX position duplicated for each UE sample
                if np.array(self.dataset[tx]["tx_pos"]).ndim == 1:
                    self.dataset_filtered["tx_pos"].extend(
                        np.repeat(self.dataset[tx]["tx_pos"][np.newaxis, :], len(indices), axis=0).tolist()
                    )
                else:
                    self.dataset_filtered["tx_pos"].extend(
                        np.repeat(self.dataset[tx]["tx_pos"], len(indices), axis=0).tolist()
                    )
        self.seed = seed
        self.total_length = len(self.dataset_filtered[list(self.dataset_filtered.keys())[0]])
        try:
            boundary = self.dataset[0]['scene'].bounding_box.bounds
            self.mins = boundary[0]
            self.maxs = boundary[1]
        except:
            boundary = self.dataset['scene'].bounding_box.bounds
            self.mins = boundary[0]
            self.maxs = boundary[1]
    def decode_interaction_to_multilabel(self, inter_code):
        """
        Convert interaction code to multi-label vector [R, D, S, T]
        Ignores repetitions: RRD -> RD, RR -> R

        Returns:
            np.array of shape (4,): binary indicators for [R, D, S, T]
            Returns [-1, -1, -1, -1] for invalid/NaN codes
        """
        if np.isnan(inter_code):
            return np.array([-1, -1, -1, -1], dtype=np.float32)

        code_str = str(int(inter_code))
        multi_label = np.zeros(4, dtype=np.float32)

        # Map: 1->R, 2->D, 3->S, 4->T
        for digit in code_str:
            if digit == '1':
                multi_label[0] = 1  # R
            elif digit == '2':
                multi_label[1] = 1  # D
            elif digit == '3':
                multi_label[2] = 1  # S
            elif digit == '4':
                multi_label[3] = 1  # T

        return multi_label

    def __getitem__(self, idx):
        prompt = []
        environment = []
        ## add building count
        # for k in ["num_materials", "bounding_box"]:
        #     print( f"{k} :", len(self.dataset_filtered[k]))
        environment.append(self.dataset_filtered["num_materials"][idx])
        environment.extend(self.dataset_filtered["bounding_box"][idx])
        environment_material_props = self.dataset_filtered["material_properties"][idx]
        

        for k in ["tx_pos", "rx_pos"]:
            # prompt.extend(self.dataset_filtered[k][idx])
            if self.normalizers and "pos" in self.apply_normalizers:
                # vals =  (self.dataset_filtered[k][idx] - self.mins)/ ( self.maxs -  self.mins)
                vals = self.dataset_filtered[k][idx]
                vals = (vals -  self.normalizers["rx_pos"]["mean"])/ self.normalizers["rx_pos"]["std"]
                prompt.extend( vals )
            
            else:
                prompt.extend( (self.dataset_filtered[k][idx] - self.mins)/ ( self.maxs -  self.mins) )



        # Sort paths based on sort_by option
        if self.sort_by == "power":
            indices = np.argsort(-np.array(self.dataset_filtered["power"][idx]))
        elif self.sort_by == "delay":
            indices = np.argsort(np.array(self.dataset_filtered["delay"][idx]))
        else:
            raise ValueError(f"Unknown sort_by option: {self.sort_by}")

        paths = []
        interactions = []  # NEW: multi-label interactions
        valid_paths = 0

        # SOS token
        paths.append([0.0, 0.0, 0.0, 0.0, 0.0])
        interactions.append([-1, -1, -1, -1])  # SOS has no interaction label

        for step_idx, indx in enumerate(indices):
            output_per_step = []
            broken = False

            for k in ["delay", "power", "phase",  "aoa_az", "aoa_el"]:
                value = self.dataset_filtered[k][idx][indx]
                if np.isnan(value):
                    value = self.pad_value
                    broken = True
                    break
                elif k == "delay":
                    value = value * 1e6  # convert to us

                elif k == "phase":
                    value = value * (np.pi/180)
                elif k in ["aoa_az", "aoa_el"]:
                    # angles in degrees -> convert to radians
                    value = value * (np.pi/180)
                elif k == "power":
                    value = value * 0.01

                if k in self.apply_normalizers:
                    value = (value - self.normalizers[k]["mean"])/ self.normalizers[k]["std"]
                output_per_step.append(value)

            if not broken:
                valid_paths += 1
                paths.append(output_per_step)

                # NEW: Extract and decode interaction
                inter_value = self.dataset_filtered["inter"][idx][indx]
                inter_label = self.decode_interaction_to_multilabel(inter_value)
                interactions.append(inter_label)

        num_paths = [valid_paths]


        return (torch.tensor(prompt, dtype=torch.float32),
                torch.tensor(paths, dtype=torch.float32),
                torch.tensor(num_paths, dtype=torch.float32) / 25.0,
                torch.tensor(interactions, dtype=torch.float32),
                torch.tensor(environment, dtype=torch.float32),
                torch.tensor(environment_material_props, dtype=torch.float32),)  # NEW

    def __len__(self):
        return self.total_length

    def collate_fn(self, batch):
        batch_prompts = torch.cat([i[0].unsqueeze(0) for i in batch], dim=0)
        batch_paths = [i[1] for i in batch]
        batch_paths = torch.nn.utils.rnn.pad_sequence(batch_paths, batch_first=True,
                                                       padding_value=self.pad_value)
        batch_num_paths = [i[2] for i in batch]
        batch_num_paths = torch.nn.utils.rnn.pad_sequence(batch_num_paths, batch_first=True,
                                                           padding_value=0)

        # NEW: collate interactions
        batch_interactions = [i[3] for i in batch]
        batch_interactions = torch.nn.utils.rnn.pad_sequence(batch_interactions, batch_first=True,
                                                             padding_value=-1)
        batch_environment = torch.cat([i[4].unsqueeze(0) for i in batch], dim=0)
        
        # batch_environment_material_props = torch.cat([i[5].unsqueeze(0) for i in batch], dim=0)
        batch_environment_material_props = torch.nn.utils.rnn.pad_sequence([i[5] for i in batch], batch_first=True,
                                                           padding_value=0)
        return batch_prompts, batch_paths, batch_num_paths, batch_interactions, batch_environment, batch_environment_material_props





class ScenarioDataset(Dataset):
    """Wrapper to add scenario identifier to each sample"""
    def __init__(self, base_dataset, scenario_name):
        self.base_dataset = base_dataset
        self.scenario_name = scenario_name
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        prompts, paths, num_paths, interactions,env, env_props = self.base_dataset[idx]
        # Add scenario identifier (optional, for tracking)
        return prompts, paths, num_paths, interactions, env, env_props
    


def create_mixed_dataloader(all_scenarios, config, train=True):
    """
    Create a single dataloader that mixes batches from all scenarios
    
    Args:
        all_scenarios: list of scenario names
        config: configuration dict with BATCH_SIZE, etc.
        train: whether to create train or validation loader
    
    Returns:
        DataLoader with mixed scenario batches
    """
    scenario_datasets = []
    pad_value = config.get('PAD_VALUE', 500)
    
    for scenario in all_scenarios:
        try:
            print(f"Loading scenario: {scenario}")
            dataset = dm.load(scenario)
            
            # Create the scenario-specific dataset
            seq_data = PreTrainMySeqDataLoader(
                dataset, 
                train=train, 
                split_by="user", 
                sort_by="power",
                pad_value=pad_value
            )
            
            # Wrap with scenario identifier
            wrapped_dataset = ScenarioDataset(seq_data, scenario)
            scenario_datasets.append(wrapped_dataset)
            
            del dataset  # Free memory
            
        except Exception as e:
            print(f"Error loading scenario {scenario}: {e}")
            continue
    
    if not scenario_datasets:
        raise ValueError("No scenarios successfully loaded!")
    
    # Concatenate all scenario datasets
    combined_dataset = ConcatDataset(scenario_datasets)
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    print(f"Number of scenarios: {len(scenario_datasets)}")
    
    # Create custom collate function
    def mixed_collate_fn(batch):
        batch_prompts = torch.stack([item[0] for item in batch])
        
        batch_paths = [item[1] for item in batch]
        batch_paths = torch.nn.utils.rnn.pad_sequence(
            batch_paths, batch_first=True, padding_value=pad_value
        )
        
        batch_num_paths = torch.stack([item[2] for item in batch])
        
        batch_interactions = [item[3] for item in batch]
        batch_interactions = torch.nn.utils.rnn.pad_sequence(
            batch_interactions, batch_first=True, padding_value=-1
        )
        batch_environment = torch.cat([i[4].unsqueeze(0) for i in batch], dim=0)
        # print("ENV?", batch[0][5].shape )
        # batch_environment_material_props = torch.cat([i[5].unsqueeze(0) for i in batch], dim=0)
        batch_environment_material_props = torch.nn.utils.rnn.pad_sequence([i[5] for i in batch], batch_first=True,
                                                           padding_value=0)
        return batch_prompts, batch_paths, batch_num_paths, batch_interactions, batch_environment, batch_environment_material_props

        
        # Optional: return scenario names for tracking
        scenario_names = [item[4] for item in batch]
        
        return batch_prompts, batch_paths, batch_num_paths, batch_interactions
    
    # Create the mixed dataloader
    mixed_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=train,  # Shuffle for training, not for validation
        collate_fn=mixed_collate_fn,
        num_workers=0,  # Adjust based on your system
        pin_memory=True
    )
    
    return mixed_loader
