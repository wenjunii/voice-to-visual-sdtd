import threading
from functools import partial

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


CONTROL_ADDRESSES = {
    "/control/gender": "gender",
    "/control/age": "age",
    "/control/visual_mode": "visual_mode",
    "/control/prompt_style": "prompt_style",
    "/control/language": "language",
    "/control/reset_scene": "reset_scene",
    "/control/request_status": "request_status",
}

CONTROL_ALIASES = {
    "gender": {
        "m": "man",
        "man": "man",
        "male": "man",
        "w": "woman",
        "woman": "woman",
        "female": "woman",
        "n": "neutral",
        "neutral": "neutral",
        "person": "neutral",
    },
    "age": {
        "1": "young",
        "young": "young",
        "2": "adult",
        "adult": "adult",
        "3": "elder",
        "elder": "elder",
        "elderly": "elder",
    },
    "visual_mode": {
        "d": "asian_american",
        "asian": "asian_american",
        "asian_american": "asian_american",
        "b": "black_brown",
        "black_brown": "black_brown",
        "black_and_brown": "black_brown",
        "x": "asian_black_brown",
        "combined": "asian_black_brown",
        "asian_black_brown": "asian_black_brown",
    },
    "prompt_style": {
        "f": "human_focus",
        "human": "human_focus",
        "human_focus": "human_focus",
        "g": "general_scene",
        "general": "general_scene",
        "general_scene": "general_scene",
    },
    "language": {
        "a": None,
        "auto": None,
        "auto_detect": None,
        "e": "en",
        "en": "en",
        "english": "en",
        "c": "zh",
        "zh": "zh",
        "chinese": "zh",
        "s": "es",
        "es": "es",
        "spanish": "es",
    },
}


def normalize_control_value(control_name, value):
    if control_name in {"reset_scene", "request_status"}:
        return True
    aliases = CONTROL_ALIASES.get(control_name)
    if aliases is None:
        raise ValueError(f"unknown control '{control_name}'")
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"invalid {control_name} value '{value}'")
    return aliases[key]


class OscControlServer:
    def __init__(self, ip, port, on_control, logger=print):
        self.ip = ip
        self.port = port
        self.on_control = on_control
        self.logger = logger
        self.server = None
        self.thread = None

    def start(self):
        if self.server is not None:
            raise RuntimeError("The OSC control server is already started")
        dispatcher = Dispatcher()
        for address, control_name in CONTROL_ADDRESSES.items():
            dispatcher.map(address, partial(self._handle, control_name))
        server = ThreadingOSCUDPServer((self.ip, self.port), dispatcher)
        thread = None
        try:
            thread = threading.Thread(
                name="voice-to-visual-osc-control",
                target=server.serve_forever,
            )
            thread.start()
        except BaseException:
            try:
                # shutdown() requires a running loop. An interrupt may arrive
                # after the native thread starts but before start() returns.
                if thread is not None and thread.ident is not None:
                    server.shutdown()
                    thread.join()
            finally:
                server.server_close()
            raise
        self.server = server
        self.thread = thread
        return server.server_address

    def stop(self):
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join()
        self.server = None
        self.thread = None

    def _handle(self, control_name, _address, *values):
        raw_value = values[0] if values else True
        try:
            value = normalize_control_value(control_name, raw_value)
            self.on_control(control_name, value)
        except Exception as exc:
            self.logger(f"OSC control request failed: {exc}")
