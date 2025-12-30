import torch
from collections import defaultdict
import numpy as np

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

        boundary = self.dataset[0]['rt_params']['raw_params']['studyarea']['boundary']['data']
        self.mins = np.array([boundary[0][0], boundary[0][1], 
                             self.dataset[0]['rt_params']['raw_params']['studyarea']['boundary']['values']['zmin']]).astype(np.float32)
        self.maxs = np.array([boundary[2][0], boundary[2][1], 
                             self.dataset[0]['rt_params']['raw_params']['studyarea']['boundary']['values']['zmax']]).astype(np.float32)

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



