from flask import Flask
from source.config import get_logger

_logger = get_logger(logger_name=__name__)

def create_app(*, config_object) -> Flask:
    """Create a flask app instance."""

    flask_app = Flask('api')
    flask_app.config.from_object(config_object)

    # import blueprints
    from source.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    _logger.debug('Application instance created')

    return flask_app
