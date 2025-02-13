class Cell:
    def __init__(self):
        self._midpoint = 0.0
        self._volume = 0.0
        # self._mass = 0.0
        # self._momentum = 0.0
        # self._energy = 0.0
        self._density = 0.0
        self._velocity = 0.0
        self._pressure = 0.0

        self._right_ngb = None
        self._surface_area = 0.0
