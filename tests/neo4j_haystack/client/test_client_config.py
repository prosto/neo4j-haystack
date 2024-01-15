import pytest
from neo4j import basic_auth

from neo4j_haystack.client import Neo4jClientConfig


@pytest.mark.unit
def test_url_is_mandatory():
    with pytest.raises(ValueError):
        Neo4jClientConfig(url=None)


@pytest.mark.unit
def test_auth_is_mandatory():
    with pytest.raises(ValueError):
        Neo4jClientConfig(username=None)

    with pytest.raises(ValueError):
        Neo4jClientConfig(password=None)

    # should not raise as `auth` credentials provided
    Neo4jClientConfig(username=None, password=None, auth=basic_auth("username", "password"))


@pytest.mark.unit
def test_client_config_to_dict():
    config = Neo4jClientConfig(
        url="bolt://localhost:7687",
        database="neo4j",
        username="username",
        password="password",
        driver_config={"connection_timeout": 60.0, "keep_alive": True},
        session_config={"fetch_size": 1000},
        transaction_config={"timeout": 10},
        use_env=False,
    )
    data = config.to_dict()

    assert data == {
        "type": "neo4j_haystack.client.neo4j_client.Neo4jClientConfig",
        "init_parameters": {
            "url": "bolt://localhost:7687",
            "database": "neo4j",
            "username": "username",
            "password": "password",
            "driver_config": {"connection_timeout": 60.0, "keep_alive": True},
            "session_config": {"fetch_size": 1000},
            "transaction_config": {"timeout": 10},
            "use_env": False,
        },
    }


@pytest.mark.unit
def test_client_config_from_dict():
    data = {
        "type": "neo4j_haystack.client.neo4j_client.Neo4jClientConfig",
        "init_parameters": {
            "url": "bolt://localhost:7687",
            "database": "neo4j",
            "username": "username",
            "password": "password",
            "driver_config": {"connection_timeout": 60.0, "keep_alive": True},
            "session_config": {"fetch_size": 1000},
            "transaction_config": {"timeout": 10},
        },
    }

    config = Neo4jClientConfig.from_dict(data)

    assert config.url == "bolt://localhost:7687"
    assert config.database == "neo4j"
    assert config.username == "username"
    assert config.password == "password"
    assert config.driver_config == {"connection_timeout": 60.0, "keep_alive": True}
    assert config.session_config == {"fetch_size": 1000}
    assert config.transaction_config == {"timeout": 10}
