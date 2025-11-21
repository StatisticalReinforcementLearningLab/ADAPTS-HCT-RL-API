from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    def __init__(self, seed: int = None):
        """
        Initialize the RL algorithm with any parameters or configurations.
        """
        pass

    @abstractmethod
    def get_action(self, group_id, state, parameters, decision_type, decision_idx) -> tuple:
        """
        Generate an action based on the given group_id, state, decision type, decision index
        and model parameters. Return the action, probability of the action.
        decision_type is the type of decision to be made.
        decision_idx is the index of the decision to be made.
        """
        pass

    @abstractmethod
    def update(self, old_params, data) -> tuple:
        """
        Update the RL algorithm with new data and/or parameters.
        """
        pass

    @abstractmethod
    def make_state(self, context) -> tuple:
        """
        Create a state representation based on the context.
        """
        pass

    @abstractmethod
    def make_reward(self, user_id, state, action, outcome) -> tuple:
        """
        Create a reward based on the user_id, state, action and outcome.
        """
        pass