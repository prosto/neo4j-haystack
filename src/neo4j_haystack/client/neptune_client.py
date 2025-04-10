import json

import boto3
from botocore.auth import SigV4Auth, _host_from_url
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from haystack import Document
from neo4j import Auth

from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore

SCHEME = "basic"
REALM = "realm"
SERVICE_NAME = "neptune-db"
DUMMY_USERNAME = "username"
HTTP_METHOD_HDR = "HttpMethod"
HTTP_METHOD = "GET"
AUTHORIZATION = "Authorization"
X_AMZ_DATE = "X-Amz-Date"
X_AMZ_SECURITY_TOKEN = "X-Amz-Security-Token"
HOST = "Host"

# ——— Neptune Endpoint ———
REGION = "us-west-2"
NEPTUNE_HOST = "neptune-cluster.cluster-cho00ee0ohft.us-west-2.neptune.amazonaws.com"
NEPTUNE_PORT = 8182
BOLT_URL = f"bolt://{NEPTUNE_HOST}:{NEPTUNE_PORT}"


class NeptuneAuthToken(Auth):
    def __init__(self, credentials: Credentials, region: str, url: str, **parameters):
        # Do NOT add "/opencypher" in the line below if you're using an engine version older than 1.2.0.0
        request = AWSRequest(method=HTTP_METHOD, url=url + "/opencypher")
        request.headers.add_header("Host", _host_from_url(request.url))
        sigv4 = SigV4Auth(credentials, SERVICE_NAME, region)
        sigv4.add_auth(request)

        auth_obj = {hdr: request.headers[hdr] for hdr in [AUTHORIZATION, X_AMZ_DATE, X_AMZ_SECURITY_TOKEN, HOST]}
        auth_obj[HTTP_METHOD_HDR] = request.method
        creds: str = json.dumps(auth_obj)

        super().__init__(SCHEME, DUMMY_USERNAME, creds, REALM, **parameters)


if __name__ == "__main__":
    boto_session = boto3.Session()
    credentials = boto_session.get_credentials().get_frozen_credentials()
    auth_token = NeptuneAuthToken(credentials, REGION, BOLT_URL)

    document_store = Neo4jDocumentStore(
        client_config=Neo4jClientConfig(
            url=BOLT_URL,
            database=None,  # Required by Neptune
            driver_config={"auth": auth_token, "encrypted": True},
        ),
        recreate_index=False,  # Not relevant for Neptune (index is created separately)
        create_index_if_missing=False,  # Should be False. Not relevant for Neptune (index is created separately)
    )

    document_store.neo4j_client.execute_write(
        "CREATE (charlie:Document {content: 'Charlie Sheen'}), (oliver:Document {content: 'Oliver Stone'})"
    )

    document_store.write_documents([Document(content="Test document")])

    # print("Filtered documents:", document_store.filter_documents())
