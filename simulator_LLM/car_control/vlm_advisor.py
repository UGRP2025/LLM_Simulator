import threading
import time
import json

class VLMAdvisor:
    """
    VLM Advisor thread.

    TODO: Implement the VLM querying and JSON parsing.
    """
    def __init__(self, hz=8, timeout_s=0.08):
        self.hz = hz
        self.timeout_s = timeout_s
        self.latest_hint = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            # TODO: Get context and query VLM
            context = self._get_context()
            
            # Placeholder for VLM response
            vlm_response_json = '''
            {
                "lane": "center",
                "speed": "normal",
                "reason": "No obstacles detected.",
                "confidence": 0.9
            }
            '''
            
            try:
                # TODO: Add schema and confidence validation
                self.latest_hint = json.loads(vlm_response_json)
            except json.JSONDecodeError:
                self.latest_hint = None

            time.sleep(1.0 / self.hz)

    def _get_context(self):
        # TODO: Aggregate context from perception and vehicle state
        return {}

    def get_latest_hint(self):
        # TODO: Add timeout logic
        return self.latest_hint
