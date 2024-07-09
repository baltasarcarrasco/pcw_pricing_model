import logging


def get_logger(log_file_path):
    """
    Configure logger
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file_path,
        level=logging.INFO,
    )
    return logging.getLogger()
