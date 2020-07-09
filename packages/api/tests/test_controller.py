import io
import json
import math
import os

from deepfake_detection.config import config as ccn_config
from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200

def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version

def test_classifier_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the neural_network_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    data_dir = os.path.abspath(os.path.join(ccn_config.DATA_FOLDER, os.pardir))
    test_dir = os.path.join(data_dir, 'sample_data')

    real_image_dir = os.path.join(test_dir, 'real')
    real_image = os.path.join(real_image_dir, '1.png')

    with open(real_image, "rb") as image_file:
        file_bytes = image_file.read()
        real_data = dict(
            file=(io.BytesIO(bytearray(file_bytes)), "1.png"),
        )

    # When
    real_response = flask_test_client.post('/predict/classifier/real',
                                      content_type='multipart/form-data',
                                      data=real_data)

    fake_image_dir = os.path.join(test_dir, 'fake')
    fake_image = os.path.join(fake_image_dir, '0.png')

    with open(fake_image, "rb") as image_file:
        file_bytes = image_file.read()
        fake_data = dict(
            file=(io.BytesIO(bytearray(file_bytes)), "0.png"),
        )

    fake_response = flask_test_client.post('/predict/classifier/fake',
                                      content_type='multipart/form-data',
                                      data=fake_data)

    # Then
    assert real_response.status_code == 200
    real_response_json = json.loads(real_response.data)
    assert real_response_json['readable_predictions']

    assert fake_response.status_code == 200
    fake_response_json = json.loads(fake_response.data)
    assert fake_response_json['readable_predictions']
