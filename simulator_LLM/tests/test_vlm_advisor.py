import pytest
import time
import json
from car_control.vlm_advisor import VLMAdvisor

# --- Test Cases ---

def test_advisor_initialization():
    """Test that the VLMAdvisor class initializes correctly."""
    advisor = VLMAdvisor()
    assert advisor.latest_hint() is None
    assert not advisor._running
    assert advisor._thread is None

def test_advisor_start_stop():
    """Test the start and stop methods for the advisor thread."""
    advisor = VLMAdvisor(hz=100) # High hz to ensure loop runs quickly
    advisor.start()
    assert advisor._running
    assert advisor._thread is not None
    assert advisor._thread.is_alive()
    time.sleep(0.05) # Give thread time to run at least once
    advisor.stop()
    assert not advisor._running
    # The thread should be joined and thus not alive anymore
    assert not advisor._thread.is_alive()

def test_valid_hint_is_stored(monkeypatch):
    """Test that a valid, high-confidence hint is correctly parsed and stored."""
    valid_response = json.dumps({
        "lane": "center", "speed": "normal",
        "reason": "test", "confidence": 0.9
    })
    # Mock the model call to always return a valid response
    monkeypatch.setattr(VLMAdvisor, '_mock_call_model', lambda self, context: valid_response)
    
    advisor = VLMAdvisor(hz=100, conf_thresh=0.8)
    advisor.start()
    time.sleep(0.05)
    advisor.stop()

    hint = advisor.latest_hint()
    assert hint is not None
    assert hint['lane'] == 'center'
    assert hint['confidence'] == 0.9

def test_low_confidence_hint_is_rejected(monkeypatch):
    """Test that a hint with confidence below the threshold is rejected."""
    low_conf_response = json.dumps({
        "lane": "center", "speed": "normal",
        "reason": "test", "confidence": 0.5
    })
    monkeypatch.setattr(VLMAdvisor, '_mock_call_model', lambda self, context: low_conf_response)
    
    advisor = VLMAdvisor(hz=100, conf_thresh=0.8)
    advisor.start()
    time.sleep(0.05)
    advisor.stop()

    assert advisor.latest_hint() is None

def test_timeout_is_handled(monkeypatch):
    """Test that a timeout (None response) from the model is handled correctly."""
    # Mock the model call to always return None, simulating a timeout
    monkeypatch.setattr(VLMAdvisor, '_mock_call_model', lambda self, context: None)
    
    advisor = VLMAdvisor(hz=100)
    advisor.start()
    # Set a valid hint first to ensure it gets cleared
    advisor._update_hint({"lane": "center"})
    time.sleep(0.05)
    advisor.stop()

    assert advisor.latest_hint() is None

def test_malformed_json_is_rejected(monkeypatch):
    """Test that a malformed JSON string is rejected."""
    malformed_response = '{"lane":"center", "speed":"normal",'
    monkeypatch.setattr(VLMAdvisor, '_mock_call_model', lambda self, context: malformed_response)
    
    advisor = VLMAdvisor(hz=100)
    advisor.start()
    time.sleep(0.05)
    advisor.stop()

    assert advisor.latest_hint() is None

def test_invalid_schema_is_rejected(monkeypatch):
    """Test that a hint missing required keys is rejected."""
    invalid_schema_response = json.dumps({
        "lane": "center", "confidence": 0.9 # Missing 'speed' and 'reason'
    })
    monkeypatch.setattr(VLMAdvisor, '_mock_call_model', lambda self, context: invalid_schema_response)
    
    advisor = VLMAdvisor(hz=100)
    advisor.start()
    time.sleep(0.05)
    advisor.stop()

    assert advisor.latest_hint() is None
