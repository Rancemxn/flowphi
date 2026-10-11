from enum import Enum
from io import BytesIO
import struct
from typing import NamedTuple, Self, TypeAlias
from flowphi.charts.model import Position, Vector


def det(a: Position, b: Position) -> float:
    return a.real * b.imag - a.imag * b.real


def intersect(line1: tuple[Position, Position], line2: tuple[Position, Position]) -> Position | None:
    dl1 = line1[0] - line1[1]
    dl2 = line2[0] - line2[1]
    xd = complex(dl1.real, dl2.real)
    yd = complex(dl1.imag, dl2.imag)
    di = det(xd, yd)
    if di == 0:
        return None
    d = complex(det(*line1), det(*line2))
    return complex(det(d, xd) / di, det(d, yd) / di)

class ScreenUtil:
    width: int
    height: int

    flick_radius: float

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.flick_radius = width * 0.1

    def visible(self, pos: Position) -> bool:
        return (0 <= pos.real <= self.width) and (0 <= pos.imag <= self.height)

    def remap(self, p: Position, rotation: Vector) -> Position:
        if self.visible(p):
            return p

        # 在直线上取一个新点q
        q = p + rotation * 1j

        # 依次求与四条边所在直线的交点
        j1 = intersect((p, q), (0, self.width))  # (0, 0) -> (W, 0)
        j2 = intersect((p, q), (0, self.height * 1j))  # (0, 0) -> (0, H)
        j3 = intersect((p, q), (self.width, complex(self.width, self.height)))  # (W, 0) -> (W, H)
        j4 = intersect((p, q), (self.height * 1j, complex(self.width, self.height)))  # (0, H) -> (W, H)

        s = 0
        cnt = 0
        if j1 is not None and (0 <= j1.real <= self.width):
            s += j1
            cnt += 1
        if j2 is not None and (0 <= j2.imag <= self.height):
            s += j2
            cnt += 1
        if j3 is not None and (0 <= j3.imag <= self.height):
            s += j3
            cnt += 1
        if j4 is not None and (0 <= j4.real <= self.width):
            s += j4
            cnt += 1

        if s == 0:
            return complex(self.width, self.height) / 2
        return s / cnt

class TouchAction(Enum):
    DOWN = 0
    UP = 1
    MOVE = 2
    CANCEL = 3
    OUTSIDE = 4
    POINTER_DOWN = 5
    POINTER_UP = 6
    HOVER_MOVE = 7

class TouchEvent(NamedTuple):
    pos: tuple[int, int]
    action: TouchAction
    pointer_id: int

class VirtualTouchEvent(NamedTuple):
    pos: Position
    action: TouchAction
    pointer_id: int

    def __str__(self) -> str:
        return f'''TouchEvent<{self.pointer_id} {self.action.name} @ ({self.pos.real:4.2f}, {self.pos.imag:4.2f})>'''

    def to_serializable(self) -> bytes:
        return struct.pack('!BIdd', self.action.value, self.pointer_id, self.pos.real, self.pos.imag)

    @classmethod
    def from_serializable(cls, data: bytes) -> Self:
        action, pointer_id, x, y = struct.unpack('!BIdd', data)
        return VirtualTouchEvent(complex(x, y), TouchAction(action), pointer_id)

RawAnswerType: TypeAlias = list[tuple[int, list[VirtualTouchEvent]]]

def dump_data(screen: ScreenUtil, ans: RawAnswerType) -> bytes:
    result = bytearray(b'PSAP' + struct.pack('!II', screen.width, screen.height))
    for ts, events in ans:
        result.extend(struct.pack('!iB', ts, len(events)))
        for event in events:
            result.extend(event.to_serializable())
    return result

def load_data(content: bytes) -> tuple[ScreenUtil, RawAnswerType]:
    reader = BytesIO(content)
    assert reader.read(4) == b'PSAP'
    width, height = struct.unpack('!II', reader.read(8))
    ans = []
    while True:
        ts_bytes = reader.read(4)
        if not ts_bytes:
            break
        (ts,) = struct.unpack('!i', ts_bytes)
        (length,) = struct.unpack('!b', reader.read(1))
        ans.append((ts, [VirtualTouchEvent.from_serializable(reader.read(21)) for _ in range(length)]))
    return ScreenUtil(width, height), ans
