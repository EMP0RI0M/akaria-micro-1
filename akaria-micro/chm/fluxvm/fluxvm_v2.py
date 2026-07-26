import torch
import torch.nn as nn
from collections import deque

class FluxVMControllerV2(nn.Module):
    """
    FluxVM V2 Controller: Differentiable adaptation of the CI-Lang stability architecture.
    Maps paper-defined mechanisms (MAAC PID, preemption triggers, scheduling barrier) to CHM.
    """
    def __init__(self, tau=1.0, alpha=0.9, gamma_P=0.1, gamma_I=0.01, gamma_D=0.05, buffer_size=20, K=2):
        super().__init__()
        # --- Paper-derived Hyperparameters ---
        self.tau = tau          # Stability threshold (D(t) > tau)
        self.alpha = alpha      # EWMA decay factor for M_t
        self.gamma_P = gamma_P  # Proportional gain
        self.gamma_I = gamma_I  # Integral gain
        self.gamma_D = gamma_D  # Derivative gain
        self.K = K              # Number of stabilization micro-steps
        
        # --- CHM Adaptation Hyperparameters ---
        self.buffer_size = buffer_size  # For historical mean M_bar
        self.history_M = deque(maxlen=buffer_size)
        
        # Recurrent state across the sequence
        self.reset_state()

    def reset_state(self):
        """Resets the controller state at the start of a sequence/episode."""
        self.D_prev = 0.0
        self.M_prev = 0.0
        self.J_prev = 0.0  # Integral of M
        self.history_M.clear()
        
    def _compute_divergence(self, h):
        """
        Paper-derived: D(t) = (1/N) * sum_i || x_i - x_bar ||^2
        CHM adaptation: Variance across the last dimension as a proxy for feature divergence.
        """
        # CHM Adaptation: using variance across the embedding dimension as divergence proxy.
        return h.var(dim=-1).mean()
        
    def forward(self, candidate, ablation_mode='E4', lambda_barrier=0.1):
        """
        candidate: the representation from the recurrent block (h_t -> candidate)
        ablation_mode: 
            'E0' = Observe (no control)
            'E2' = P/D + threshold (no memory trigger/integral)
            'E3' = MAAC/PID + memory trigger (no iterative barrier)
            'E4' = MAAC/PID + fixed-K barrier
        (Note: E1 is the old exponential controller and would be tested via the V1 adapter).
        """
        # Ensure D_prev is on correct device
        if isinstance(self.D_prev, float):
            self.D_prev = torch.tensor(0.0, device=candidate.device)
            self.M_prev = torch.tensor(0.0, device=candidate.device)
            self.J_prev = torch.tensor(0.0, device=candidate.device)

        # --- 1. PROBE (Paper-derived) ---
        D_t = self._compute_divergence(candidate)
        
        # MAAC Memory Dynamics (Paper Eq: M_t = alpha * M_prev + (1-alpha) * D_t)
        M_t = self.alpha * self.M_prev + (1 - self.alpha) * D_t
        # Integral of memory (CHM adaptation of continuous integral)
        J_t = self.J_prev + M_t
        # Derivative (finite difference)
        dD_t = D_t - self.D_prev
        
        # Historical Mean
        if len(self.history_M) > 0:
            M_bar = sum(self.history_M) / len(self.history_M)
        else:
            M_bar = M_t.item()
            
        V_prev = 0.5 * (self.D_prev**2) + 0.5 * (self.M_prev**2)
        
        if ablation_mode == 'E0':
            self._update_state(D_t, M_t, J_t)
            return candidate, torch.tensor(0.0, device=candidate.device), torch.tensor(1.0, device=candidate.device)
            
        # --- 2. TRIGGER (CHM Adaptation of discrete branching) ---
        g_D = torch.relu(D_t - self.tau)
        
        if ablation_mode in ['E3', 'E4']:
            g_M = torch.relu(M_t - M_bar)
        else:
            g_M = torch.tensor(0.0, device=candidate.device)
            
        # Differentiable gate: non-zero if either trigger is active
        # This preserves the rule: stable state => controller does nothing.
        G_t = torch.tanh(g_D + g_M) 
        
        # --- 3. CONTROL (Paper-derived PID) ---
        if ablation_mode == 'E2':
            # P/D + threshold
            I_t = G_t * (self.gamma_P * (D_t - self.tau) + self.gamma_D * dD_t)
        else:
            # MAAC/PID
            I_t = G_t * (self.gamma_P * (D_t - self.tau) + self.gamma_I * J_t + self.gamma_D * dD_t)
            
        # --- 4. MICRO-BARRIER (CHM Adaptation of Paper's strict clock-advance) ---
        x = candidate
        K_steps = self.K if ablation_mode == 'E4' else 1
        
        for k in range(K_steps):
            # Do NOT advance M/J/history here (Paper constraint).
            
            # CHM Adaptation: centroid correction operator (pull towards mean)
            x_mean = x.mean(dim=-1, keepdim=True)
            correction = - (x - x_mean)
            
            # Differentiable intervention
            x_new = x + I_t * correction
            
            x = x_new
            
        # --- 5. COMMIT & ACCEPTANCE (Paper-derived V(X) barrier) ---
        D_final = self._compute_divergence(x)
        # Note: Paper says V(X) = 0.5 D^2 + 0.5 M^2. 
        V_corrected = 0.5 * (D_final**2) + 0.5 * (M_t**2)
        
        delta_V = V_corrected - V_prev
        barrier_pass = (delta_V < 0).float()
        
        # CHM Adaptation: Auxiliary loss to encourage passing the barrier
        L_barrier = lambda_barrier * torch.relu(delta_V) if ablation_mode == 'E4' else torch.tensor(0.0, device=candidate.device)
        
        # --- 6. UPDATE RECURRENT CONTROLLER STATE ---
        self._update_state(D_t, M_t, J_t)
        
        return x, L_barrier, barrier_pass
        
    def _update_state(self, D_t, M_t, J_t):
        self.D_prev = D_t.detach()
        self.M_prev = M_t.detach()
        self.J_prev = J_t.detach()
        self.history_M.append(self.M_prev.item())
