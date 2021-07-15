import base64
import os
import time
from datetime import datetime, timedelta
from getpass import getpass

import requests
from huggingface_hub import HfApi

from hivemind.proto.auth_pb2 import AccessToken
from hivemind.utils.auth import TokenAuthorizerBase
from hivemind.utils.crypto import RSAPublicKey
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class NonRetriableError(Exception):
    pass


def call_with_retries(func, n_retries=10, initial_delay=1.0):
    for i in range(n_retries):
        try:
            return func()
        except NonRetriableError:
            raise
        except Exception as e:
            if i == n_retries - 1:
                raise

            delay = initial_delay * (2 ** i)
            logger.warning(f'Failed to call `{func.__name__}` with exception: {e}. Retrying in {delay:.1f} sec')
            time.sleep(delay)


class InvalidCredentialsError(NonRetriableError):
    pass


class NotInAllowlistError(NonRetriableError):
    pass


class HuggingFaceAuthorizer(TokenAuthorizerBase):
    _AUTH_SERVER_URL = 'https://collaborative-training-auth.huggingface.co'

    def __init__(self, experiment_id: int, username: str, password: str):
        super().__init__()

        self.experiment_id = experiment_id
        self.username = username
        self.password = password

        self._authority_public_key = None
        self.coordinator_ip = None
        self.coordinator_port = None

        self._hf_api = HfApi()

    async def get_token(self) -> AccessToken:
        """
        Hivemind calls this method to refresh the token when necessary.
        """

        self.join_experiment()
        return self._local_access_token

    def join_experiment(self) -> None:
        call_with_retries(self._join_experiment)

    def _join_experiment(self) -> None:
        try:
            token = self._hf_api.login(self.username, self.password)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:  # Unauthorized
                raise InvalidCredentialsError()
            raise

        try:
            url = f'{self._AUTH_SERVER_URL}/api/experiments/join/{self.experiment_id}/'
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.put(url, headers=headers, json={
                'experiment_join_input': {
                    'peer_public_key': self.local_public_key.to_bytes().decode(),
                },
            })

            response.raise_for_status()
            response = response.json()

            self._authority_public_key = RSAPublicKey.from_bytes(response['auth_server_public_key'].encode())
            self.coordinator_ip = response['coordinator_ip']
            self.coordinator_port = response['coordinator_port']

            token_dict = response['hivemind_access']
            access_token = AccessToken()
            access_token.username = token_dict['username']
            access_token.public_key = token_dict['peer_public_key'].encode()
            access_token.expiration_time = str(datetime.fromisoformat(token_dict['expiration_time']))
            access_token.signature = token_dict['signature'].encode()
            self._local_access_token = access_token
            logger.info(f'Access for user {access_token.username} '
                        f'has been granted until {access_token.expiration_time} UTC')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:  # Unauthorized
                raise NotInAllowlistError()
            raise
        finally:
            self._hf_api.logout(token)

    def is_token_valid(self, access_token: AccessToken) -> bool:
        data = self._token_to_bytes(access_token)
        if not self._authority_public_key.verify(data, access_token.signature):
            logger.exception('Access token has invalid signature')
            return False

        try:
            expiration_time = datetime.fromisoformat(access_token.expiration_time)
        except ValueError:
            logger.exception(
                f'datetime.fromisoformat() failed to parse expiration time: {access_token.expiration_time}')
            return False
        if expiration_time.tzinfo is not None:
            logger.exception(f'Expected to have no timezone for expiration time: {access_token.expiration_time}')
            return False
        if expiration_time < datetime.utcnow():
            logger.exception('Access token has expired')
            return False

        return True

    _MAX_LATENCY = timedelta(minutes=1)

    def does_token_need_refreshing(self, access_token: AccessToken) -> bool:
        expiration_time = datetime.fromisoformat(access_token.expiration_time)
        return expiration_time < datetime.utcnow() + self._MAX_LATENCY

    @staticmethod
    def _token_to_bytes(access_token: AccessToken) -> bytes:
        return f'{access_token.username} {access_token.public_key} {access_token.expiration_time}'.encode()


def authorize_with_huggingface() -> HuggingFaceAuthorizer:
    while True:
        experiment_id = os.getenv('HF_EXPERIMENT_ID')
        if experiment_id is None:
            experiment_id = input('HuggingFace experiment ID: ')

        username = os.getenv('HF_USERNAME')
        if username is None:
            while True:
                username = input('HuggingFace username: ')
                if '@' not in username:
                    break
                print('Please enter your Huggingface _username_ instead of the email address!')

        password = os.getenv('HF_PASSWORD')
        if password is None:
            password = getpass('HuggingFace password: ')

        authorizer = HuggingFaceAuthorizer(experiment_id, username, password)
        try:
            authorizer.join_experiment()
            return authorizer
        except InvalidCredentialsError:
            print('Invalid username or password, please try again')
        except NotInAllowlistError:
            print('This account is not specified in the allowlist. '
                  'Please ask a moderator to add you to the allowlist and try again')
