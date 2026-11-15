import heapq


def astar(grid, start, goal):
    """Find a 4-connected path through cells marked '.'."""
    rows = len(grid)
    cols = len(grid[0])

    def valid(cell):
        r, c = cell
        return (
            0 <= r < rows
            and 0 <= c < cols
            and grid[r][c] != "#"
        )

    frontier = [(0, start)]
    came_from = {start: None}
    cost = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        r, c = current
        for nxt in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if not valid(nxt):
                continue

            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + abs(goal[0]-nxt[0]) + abs(goal[1]-nxt[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        raise ValueError(f"No path from {start} to {goal}")

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]

    return list(reversed(path))
