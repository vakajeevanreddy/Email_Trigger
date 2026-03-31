def grade_step(task, action, step_count):
    reward = 0.0
    done = False
    info = ""

    # Category scoring
    if action.category == task["expected_category"]:
        reward += 0.4
    else:
        reward -= 0.2

    # Action scoring
    if action.action_type == task["expected_action"]:
        reward += 0.4
    else:
        reward -= 0.2

    # Response scoring (for hard task)
    if action.response:
        if len(action.response) > 10:
            reward += 0.2
        else:
            reward -= 0.1

    # Penalty for too many steps
    if step_count > 3:
        reward -= 0.3

    # Done condition
    if reward >= 0.7 or step_count >= 3:
        done = True

    info = f"Reward computed: {reward}"

    return reward, done, info