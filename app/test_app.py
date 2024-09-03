"""Test script for Flask application."""
# pylint: disable=redefined-outer-name

import pytest
from bs4 import BeautifulSoup
from app import app


@pytest.fixture
def client():
    """App testing"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    """Test if home page is loading correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'TEXT SUMMARIZATION' in response.data  # Check if title is here


def test_summarization_success(client):
    """Test if summarization task works with a valid input"""
    input_text = "This is a long text that needs to be summarized by the model."  # noqa: E501
    response = client.post('/text-summarization',
                           data={"inputtext_": input_text})
    parsed_response = BeautifulSoup(response.data, 'html.parser')

    assert response.status_code == 200
    assert hasattr(parsed_response, "p")  # Check if there is a response...
    assert len(parsed_response.p.get_text()) > 0  # ... that is not empty


def test_summarization_empty_input(client):
    """Test if error management works when input is empty"""
    response = client.post('/text-summarization', data={"inputtext_": ""})
    parsed_response = BeautifulSoup(response.data, 'html.parser')

    assert response.status_code == 200
    assert ("ERROR: No inference made. Input text can't be empty!"
            in parsed_response.p.get_text())  # Check what the response


def test_summarization_large_input(client):
    """Test summarization task with a very long input"""
    large_input_text = "This is a very large text. " * 1000
    response = client.post('/text-summarization',
                           data={"inputtext_": large_input_text})
    parsed_response = BeautifulSoup(response.data, 'html.parser')

    assert response.status_code == 200
    assert hasattr(parsed_response, "p")  # Check if there is a response...
    assert len(parsed_response.p.get_text()) > 0  # ... that is not empty
