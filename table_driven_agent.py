class TableDrivenAgent:
    def __init__(self, table):
        self.percepts = []  # Stores the sequence of percepts
        self.table = table  # Table of actions, indexed by percept sequences

    def lookup(self, percepts):
        # Convert percept sequence to a tuple (because lists are not hashable)
        percepts_tuple = tuple(percepts)
        # Lookup the action corresponding to the percept sequence
        return self.table.get(percepts_tuple, "No action found")

    def perceive(self, percept):
        # Append the new percept to the list of percepts
        self.percepts.append(percept)
        # Find the appropriate action from the table based on the percepts
        action = self.lookup(self.percepts)
        return action


# Example of table where each percept sequence maps to an action
action_table = {
    (): "Do nothing",
    ("see wall",): "Turn left",
    ("see wall", "see door"): "Open door",
    ("see wall", "see window"): "Look through window"
}

# Creating the agent
agent = TableDrivenAgent(action_table)

# Simulating percepts and getting actions
print(agent.perceive("see wall"))        # Output: Turn left
print(agent.perceive("see door"))        # Output: Open door
