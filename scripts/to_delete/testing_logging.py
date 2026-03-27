import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    log.info("This SHOULD go into main.log")
    print("This will NOT go into main.log (unless configured)")

if __name__ == "__main__":
    main()