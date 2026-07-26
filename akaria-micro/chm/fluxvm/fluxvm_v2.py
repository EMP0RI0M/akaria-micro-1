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
        CHM adaptation: Variance across the stream dimension (dim=2) to measure expert/stream divergence.
        """
        if h.dim() == 4:
            return h.var(dim=2, unbiased=False).mean()
        return h.var(dim=-1).mean()
        
    def forward(self, candidate, ablation_mode='E4', lambda_barrier=0.1):
        # Ensure D_prev is on correct device
        if isinstance(self.D_prev, float):
            self.D_prev = torch.tensor(0.0, device=candidate.device)
            self.M_prev = torch.tensor(0.0, device=candidate.device)
            self.J_prev = torch.tensor(0.0, device=candidate.device)

        # --- 1. PROBE (Paper-derived) ---
        D_t = self._compute_divergence(candidate)
        
        # MAAC Memory Dynamics
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
        
        telemetry = {
            "D_t": D_t, "D_prev": self.D_prev, "delta_D": dD_t,
            "M_t": M_t, "historical_M": torch.tensor(M_bar, device=D_t.device),
            "V_before": V_prev
        }
        
        if ablation_mode == 'E0':
            self._update_state(D_t, M_t, J_t)
            telemetry.update({
                "g_D": 0.0, "g_M": 0.0, "G_t": 0.0, "P_term": 0.0, "I_term": 0.0, "D_term": 0.0,
                "I_t": 0.0, "V_after": V_prev, "delta_V": 0.0, "barrier_pass": 1.0, "L_barrier": 0.0
            })
            return candidate, telemetry
            
        # --- 2. TRIGGER (CHM Adaptation of discrete branching) ---
        g_D = torch.relu(D_t - self.tau)
        
        if ablation_mode in ['E3', 'E4']:
            g_M = torch.relu(M_t - M_bar)
        else:
            g_M = torch.tensor(0.0, device=candidate.device)
            
        G_t = torch.tanh(g_D + g_M) 
        
        # --- 3. CONTROL (Paper-derived PID) ---
        P_term = self.gamma_P * (D_t - self.tau)
        D_term = self.gamma_D * dD_t
        I_term = self.gamma_I * J_t if ablation_mode != 'E2' else torch.tensor(0.0, device=candidate.device)
        
        I_t = G_t * (P_term + I_term + D_term)
        
        # CHM Adaptation: Anti-windup. Clamp I_t to [0, 1] to prevent centroid overshoot and divergence explosion.
        I_t = torch.clamp(I_t, min=0.0, max=1.0)
            
        # --- 4. MICRO-BARRIER (CHM Adaptation of Paper's strict clock-advance) ---
        x = candidate
        K_steps = self.K if ablation_mode == 'E4' else 1
        
        for k in range(K_steps):
            # CHM Adaptation: centroid correction operator across streams
            if x.dim() == 4:
                x_mean = x.mean(dim=2, keepdim=True)
            else:
                x_mean = x.mean(dim=-1, keepdim=True)
                
            correction = - (x - x_mean)
            x = x + I_t * correction
            
        # --- 5. COMMIT & ACCEPTANCE (Paper-derived V(X) barrier) ---
        D_final = self._compute_divergence(x)
        V_corrected = 0.5 * (D_final**2) + 0.5 * (M_t**2)
        delta_V = V_corrected - V_prev
        
        # Verify barrier_pass semantics: only pass if V reduced AND D_final <= tau (state is not pathological)
        barrier_pass = ((delta_V < 0) & (D_final <= self.tau)).float()
        
        L_barrier = lambda_barrier * torch.relu(delta_V) if ablation_mode == 'E4' else torch.tensor(0.0, device=candidate.device)
        
        # Populate detailed telemetry
        telemetry.update({
            "g_D": g_D, "g_M": g_M, "G_t": G_t,
            "P_term": P_term, "I_term": I_term, "D_term": D_term, "I_t": I_t,
            "V_after": V_corrected, "delta_V": delta_V,
            "barrier_pass": barrier_pass, "L_barrier": L_barrier
        })
        
        # --- 6. UPDATE RECURRENT CONTROLLER STATE ---
        self._update_state(D_t, M_t, J_t)
        
        return x, telemetry
        
    def _update_state(self, D_t, M_t, J_t):
        self.D_prev = D_t.detach()
        self.M_prev = M_t.detach()
        self.J_prev = J_t.detach()
        self.history_M.append(self.M_prev.item())
