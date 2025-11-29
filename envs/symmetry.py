import torch

class T1Symmetry:
    def __init__(self, env):
        self.device = env.device
        self.num_dofs = env.num_dofs   # Should be 21
        self.num_obs = env.num_obs     # Should be 74
        self.num_acts = env.num_actions # Should be 21
        self.num_priv = env.num_privileged_obs
        
        self.dof_names = env.dof_names
        
        self.left_dof_ids = []
        self.right_dof_ids = []
        self.waist_dof_ids = []
        self.neg_dof_ids = [] # Indices relative to the DOF list (0-20)

        for i, name in enumerate(self.dof_names):
            if "Left" in name:
                self.left_dof_ids.append(i)
            elif "Right" in name:
                self.right_dof_ids.append(i)
            else:
                self.waist_dof_ids.append(i)
            
            # 2. Find Axis that need Sign Flipping
            # Rules: Roll (X) and Yaw (Z) flip. Pitch (Y) stays.
            # "Waist" is typically Z-axis (Yaw) -> Flip
            if "Roll" in name or "Yaw" in name or "Waist" in name:
                self.neg_dof_ids.append(i)

        # Sanity Check
        assert len(self.left_dof_ids) == len(self.right_dof_ids), "Mismatch in Left/Right joints!"
        
        # ==========================================================
        # 2. BUILD ACTION MATRIX (21 x 21)
        # ==========================================================
        # Start with Identity
        mat_act = torch.eye(self.num_acts, device=self.device)
        flip_act = torch.ones(self.num_acts, device=self.device)

        # Apply Swaps
        for l, r in zip(self.left_dof_ids, self.right_dof_ids):
            mat_act[l, l] = 0
            mat_act[r, r] = 0
            mat_act[l, r] = 1
            mat_act[r, l] = 1
            
        # Apply Negations
        flip_act[self.neg_dof_ids] = -1
        
        self.act_transform = torch.matmul(mat_act, torch.diag(flip_act))

        # ==========================================================
        # 3. BUILD OBSERVATION MATRIX (74 x 74)
        # ==========================================================
        # We build this block by block based on your _compute_observations
        
        # Block 1: Gravity (3) [x, y, z] -> Flip Y
        # Block 2: Base Ang Vel (3) [x, y, z] -> Flip X (Roll) and Z (Yaw)
        # Block 3: Commands (3) [vx, vy, w_z] -> Flip vy and w_z
        # Block 4: Gait (2) [cos, sin] -> Negate both (Phase shift by PI)
        # Block 5: DOF Pos (21) -> Use Act Matrix logic
        # Block 6: DOF Vel (21) -> Use Act Matrix logic
        # Block 7: Last Act (21) -> Use Act Matrix logic

        # --- A. Create the Diagonal Flip Vector (Signs) ---
        obs_flip = torch.ones(self.num_obs, device=self.device)
        
        # 1. Gravity (Idx 0-2): Flip Y (Idx 1)
        obs_flip[1] = -1 
        
        # 2. Base Ang Vel (Idx 3-5): Flip X(3) and Z(5)
        obs_flip[3] = -1
        obs_flip[5] = -1
        
        # 3. Commands (Idx 6-8): Flip Vy(7) and Wz(8)
        obs_flip[7] = -1
        obs_flip[8] = -1
        
        # 4. Gait (Idx 9-10): Negate both to swap legs
        obs_flip[9] = -1
        obs_flip[10] = -1
        
        # 5, 6, 7. DOFs (Pos, Vel, PrevAction)
        # We need to apply the joint negation logic 3 times
        start_indices = [11, 11+21, 11+21+21] # 11, 32, 53
        for start_idx in start_indices:
            for rel_idx in self.neg_dof_ids:
                obs_flip[start_idx + rel_idx] = -1

        self.obs_flip_mat = torch.diag(obs_flip)

        # --- B. Create the Permutation Matrix (Swaps) ---
        self.obs_perm_mat = torch.eye(self.num_obs, device=self.device)
        
        # Apply Left<->Right swaps for the 3 DOF blocks
        for start_idx in start_indices:
            for l_rel, r_rel in zip(self.left_dof_ids, self.right_dof_ids):
                l = start_idx + l_rel
                r = start_idx + r_rel
                
                self.obs_perm_mat[l, l] = 0
                self.obs_perm_mat[r, r] = 0
                self.obs_perm_mat[l, r] = 1
                self.obs_perm_mat[r, l] = 1

        # Combine
        self.obs_transform = torch.matmul(self.obs_perm_mat, self.obs_flip_mat)
        # ==========================================================
        # 4. BUILD PRIVILEGED OBS MATRIX (14 x 14)
        # ==========================================================
        # 0: CoM X
        # 1: CoM Y      <- FLIP
        # 2: CoM Z
        # 3: Mass
        # 4: Lin Vel X
        # 5: Lin Vel Y  <- FLIP
        # 6: Lin Vel Z
        # 7: Height
        # 8: Force X
        # 9: Force Y    <- FLIP
        # 10: Force Z
        # 11: Torque X  <- FLIP (Roll)
        # 12: Torque Y
        # 13: Torque Z  <- FLIP (Yaw)

        priv_flip = torch.ones(self.num_priv, device=self.device)
        neg_priv_ids = [1, 5, 9, 11, 13]
        priv_flip[neg_priv_ids] = -1
        
        # No Permutation needed (Identity matrix), just diagonal flips
        self.priv_transform = torch.diag(priv_flip)

    def mirror_act(self, act):
        # Expects shape (Batch, 21)
        return torch.matmul(self.act_transform, act.unsqueeze(-1)).squeeze(-1)

    def mirror_obs(self, obs):
        # Expects shape (Batch, 74)
        return torch.matmul(self.obs_transform, obs.unsqueeze(-1)).squeeze(-1)
    
    def mirror_priv(self, priv):
        # Expects shape (Batch, 14)
        return torch.matmul(self.priv_transform, priv.unsqueeze(-1)).squeeze(-1)