import logging, sys

def setup_logging(level='INFO'):
    """configure structured logging for the entire project"""
    numeric_level = getattr(logging, level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt='%Y-%m-%d %H:%M:%S')
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    #silence noisy third-party loggers
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def get_logger(name):
    """shorthand to get a named logger"""
    return logging.getLogger(name)