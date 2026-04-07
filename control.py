import numpy as np

class Controller:
    def __init__(self, params=None):
        # 8 parameters for ant joints
        if params is None:
            self.params = np.random.uniform(-1, 1, 8)
        else:
            self.params = np.array(params)
    
    def get_action(self, joint_angles):
        """
        Takes 8 joint angles and returns 8 output angles.
        For simplicity, use a linear mapping with params as weights.
        """
        # joint_angles: np.array of 8 floats
        # Return target angles: params * joint_angles + bias (but params as simple offset for now)
        return np.clip(self.params, -1, 1)  # Simple fixed output, can be extended
    
    def mutate(self, mutation_rate=0.1):
        """Mutate parameters slightly"""
        noise = np.random.normal(0, mutation_rate, len(self.params))
        self.params += noise
        # Clip to bounds
        self.params = np.clip(self.params, -1, 1)
    
    def copy(self):
        """Create a copy"""
        return Controller(self.params.copy())
    
    def __str__(self):
        return f"Controller(params={self.params})"