class Cell:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.intersects = []

    def set_intersects(self, intersects):
        row_y = intersects[0][1]
        row = []
        for i in range(len''ntersects)):
            if i == len(intersects) - 1:
                row.append(intersects[i])
                self.intersects.append(row)
                break

            row.append(intersects[i])

            # If the next joint has a new y-coordinate,
            # start a new row.
            if intersects[i + 1][1] != row_y:
                self.intersects.append(row)
                row_y = intersects[i + 1][1]
                row = []

    def get_area(self):
        return self.w * self.h


