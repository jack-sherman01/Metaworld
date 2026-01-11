"""
Task descriptions for MetaWorld MT10 benchmark.
Each task name is mapped to a detailed description of the task objective.
"""

MT10_TASK_DESCRIPTIONS = {
    "reach-v3": {
        "description": "Move the end-effector of robot to reach a target position marked by a puck.",
        "objective": "Position the robot gripper at the target location.",
        "success_criteria": "Gripper is within a small threshold distance of the target puck.",
        "difficulty": "Easy"
    },
    
    "push-v3": {
        "description": "Push a puck to a goal position on the table.",
        "objective": "Use the gripper to push the puck to the target location.",
        "success_criteria": "Puck reaches the goal position.",
        "difficulty": "Medium"
    },
    
    "pick-place-v3": {
        "description": "Pick up a puck and place it at a goal position.",
        "objective": "Grasp the puck, lift it, and place it at the target location.",
        "success_criteria": "Puck is placed at the goal position.",
        "difficulty": "Medium"
    },
    
    "door-open-v3": {
        "description": "Open a door by pulling the door handle.",
        "objective": "Grasp the door handle and pull it to open the door.",
        "success_criteria": "Door is opened beyond a certain angle threshold.",
        "difficulty": "Hard"
    },
    
    "drawer-open-v3": {
        "description": "Open a drawer by pulling the drawer handle.",
        "objective": "Grasp the drawer handle and pull it outward to open the drawer.",
        "success_criteria": "Drawer is pulled out beyond a certain distance threshold.",
        "difficulty": "Medium"
    },
    
    "drawer-close-v3": {
        "description": "Close an open drawer by pushing it.",
        "objective": "Push the drawer handle inward to close the drawer.",
        "success_criteria": "Drawer is pushed in and fully closed.",
        "difficulty": "Medium"
    },
    
    "button-press-topdown-v3": {
        "description": "Press a button from a top-down approach.",
        "objective": "Move the gripper downward to press the button.",
        "success_criteria": "Button is pressed down beyond a certain threshold.",
        "difficulty": "Easy"
    },
    
    "peg-insert-side-v3": {
        "description": "Insert a peg into a hole from the side.",
        "objective": "Pick up the peg and insert it into the hole from a horizontal angle.",
        "success_criteria": "Peg is inserted into the hole.",
        "difficulty": "Hard"
    },
    
    "window-open-v3": {
        "description": "Open a window by pushing the window handle upward.",
        "objective": "Grasp the window handle and push it up to open the window.",
        "success_criteria": "Window is opened beyond a certain angle threshold.",
        "difficulty": "Hard"
    },
    
    "window-close-v3": {
        "description": "Close an open window by pushing the window handle downward.",
        "objective": "Grasp the window handle and push it down to close the window.",
        "success_criteria": "Window is fully closed.",
        "difficulty": "Hard"
    }
}


def get_task_description(task_name: str) -> dict:
    """
    Get the description for a specific task.
    
    Args:
        task_name: Name of the task (e.g., "reach-v3")
    
    Returns:
        Dictionary containing task description, objective, success criteria, and difficulty
    """
    return MT10_TASK_DESCRIPTIONS.get(task_name, {
        "description": "Unknown task",
        "objective": "N/A",
        "success_criteria": "N/A",
        "difficulty": "N/A"
    })


def print_all_tasks():
    """Print all MT10 tasks with their descriptions."""
    print("=" * 80)
    print("MetaWorld MT10 Benchmark - Task Descriptions")
    print("=" * 80)
    
    for i, (task_name, info) in enumerate(MT10_TASK_DESCRIPTIONS.items(), 1):
        print(f"\n{i}. {task_name.upper()}")
        print(f"   Difficulty: {info['difficulty']}")
        print(f"   Description: {info['description']}")
        print(f"   Objective: {info['objective']}")
        print(f"   Success: {info['success_criteria']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_all_tasks()