import logging

def setup_logger(
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(levelname)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    filename: str = None
) -> None:
    """
    Configures the root logger with basic settings.
    Call this once at the start of your program.
    """
    if filename:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, filename=filename)
    else:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Returns a logger instance for the given module name.
    """
    return logging.getLogger(name)
