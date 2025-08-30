import threading
import time
import json
import base64
from typing import Optional, Dict, Any

import cv2
from cv_bridge import CvBridge
from openai import OpenAI, APIError, APITimeoutError
from rclpy.node import Node
from sensor_msgs.msg import Image

# --- Type Aliases ---
HintDict = Dict[str, Any]

class VLMAdvisor:
    """
    Manages asynchronous calls to a VLM to get driving advice.
    Operates in a separate thread and provides safety guards for the VLM output.
    """
    def __init__(self, node: Node, params: Dict[str, Any]):
        """
        Args:
            node: The ROS2 node to use for subscriptions and parameters.
            params: A dictionary of parameters for the VLM advisor.
        """
        self.node = node
        self.params = params
        self.bridge = CvBridge()
        self.vlm_connected = False

        # Get parameters
        self.timeout_s = self.params['timeout_s']
        self.conf_thresh = self.params['conf_thresh']
        self.jpeg_w = self.params['jpeg_w']
        self.jpeg_h = self.params['jpeg_h']
        self.jpeg_q = self.params['jpeg_q']

        # OpenAI client setup
        self.client = OpenAI(base_url=self.params['api_base'], api_key=self.params['api_key'])

        self._latest_hint: Optional[HintDict] = None
        self._latest_image: Optional[Image] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_log_time = 0

        # Subscribe to the camera topic
        self.node.create_subscription(
            Image,
            self.params['camera_topic'],
            self.image_callback,
            1
        )
        
        # Check connection to VLM server
        self._check_vlm_connection()

    def _check_vlm_connection(self):
        """Checks if the VLM server is reachable."""
        try:
            self.client.models.list(timeout=5.0) # Short timeout for connection check
            self.vlm_connected = True
            self.node.get_logger().info("[VLM] Successfully connected to VLM server.")
        except (APIError, APITimeoutError) as e:
            self.node.get_logger().error(f"[VLM] Connection failed: {e}. Running without VLM hints.")
            self.vlm_connected = False

    def image_callback(self, msg: Image):
        """Stores the latest image message."""
        with self._lock:
            self._latest_image = msg

    def start(self):
        """Starts the advisor thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._advisor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the advisor thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def latest_hint(self) -> Optional[HintDict]:
        """Returns the latest valid hint received from the VLM."""
        with self._lock:
            return self._latest_hint.copy() if self._latest_hint else None

    def _update_hint(self, hint: Optional[HintDict]):
        """Thread-safely updates the latest hint."""
        with self._lock:
            self._latest_hint = hint

    def _advisor_loop(self):
        """The main loop for the advisor thread."""
        while self._running:
            if not self.vlm_connected:
                if time.time() - self._last_log_time > 5.0: # Log every 5 seconds
                    self.node.get_logger().warn("[VLM] Not connected, skipping hint generation.")
                    self._last_log_time = time.time()
                time.sleep(1.0)
                continue

            with self._lock:
                image = self._latest_image

            if image is None:
                time.sleep(0.1) # Wait for the first image
                continue

            context = self._build_context()
            request_time = time.time()
            json_str = self.call_hf_qwen_model(context, image)
            validated_hint = self._parse_and_validate(json_str)

            if validated_hint:
                validated_hint['timestamp'] = request_time
            
            self._update_hint(validated_hint)

            # No sleep here, run the next request immediately

    def _build_context(self) -> str:
        """Builds the text context/prompt for the VLM."""
        # This can be expanded with more dynamic data from the vehicle
        return """
        TASK: Pick a lane (inner, center, or outer) and speed (slow, normal, or fast) for a racing scenario.
        RULES: Prioritize safety and speed. Avoid collisions. Stay on the track.
        OUTPUT_JSON: Respond with a single JSON object. The "confidence" key must be a float between 0.0 and 1.0.
        The JSON object must contain the following keys: "lane", "speed", "reason", and "confidence".
        """

    def _encode_image(self, image_msg: Image) -> str:
        """Encodes a ROS Image message to a base64 string after JPEG compression."""
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        resized_image = cv2.resize(cv_image, (self.jpeg_w, self.jpeg_h))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q]
        _, buffer = cv2.imencode('.jpg', resized_image, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    def _parse_and_validate(self, json_str: Optional[str]) -> Optional[HintDict]:
        """Parses and validates the JSON string from the VLM."""
        if json_str is None:
            return None
        try:
            # The model might return a string with ```json ... ```, so we extract it.
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            data = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            self.node.get_logger().warn(f"[VLM] Failed to parse JSON: {json_str}")
            return None

        # Schema validation
        required_keys = ["lane", "speed", "reason", "confidence"]
        if not all(key in data for key in required_keys):
            self.node.get_logger().warn(f"[VLM] JSON missing required keys: {data}")
            return None
        
        if not isinstance(data.get('confidence'), (int, float)):
            self.node.get_logger().warn(f"[VLM] Invalid confidence type in JSON: {data}")
            return None

        # Confidence threshold validation
        if data['confidence'] < self.conf_thresh:
            return None

        return data

    def call_hf_qwen_model(self, context: str, image: Image) -> Optional[str]:
        """
        Calls the HuggingFace VL model using an openai-compatible API endpoint.
        """
        base64_image = self._encode_image(image)
        content = [
            {"type": "text", "text": context},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.params['model'],
                messages=[{"role": "user", "content": content}],
                max_tokens=self.params['max_tokens'],
                temperature=self.params['temperature'],
                timeout=self.timeout_s,
                stream=False
            )
            return response.choices[0].message.content
        except (APIError, APITimeoutError) as e:
            self.node.get_logger().error(f"[VLM ERROR] {e}")
            return None
