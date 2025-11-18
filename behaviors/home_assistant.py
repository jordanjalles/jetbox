"""Home Assistant behavior for smart home control.
Now uses @tool decorator for automatic tool registration!"""

import json
import logging
import os
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool
from behaviors.rule_of_two_types import RuleOfTwoProperty

logger = logging.getLogger(__name__)


class HomeAssistantBehavior(AgentBehavior):
    """Behavior for controlling Home Assistant smart home devices.

    Provides tools to:
    - List all devices
    - Get device state
    - Control devices (turn on/off, set brightness, adjust climate, etc.)
    - List and trigger automations
    - Get area/room information

    Security: [BC] - Sensitive access (home state) + External action (controls devices)
    Device states reveal occupancy, routines, and other sensitive information.
    """

    rule_of_two_properties = {
        RuleOfTwoProperty.SENSITIVE_ACCESS,  # Home device states are sensitive
        RuleOfTwoProperty.EXTERNAL_ACTION     # Controls physical devices
    }

    def __init__(
        self,
        api_url: Optional[str] = None,
        access_token: Optional[str] = None,
        cache_ttl: int = 30,
    ):
        """Initialize Home Assistant behavior.

        Args:
            api_url: Home Assistant API URL (default: auto-load from secrets or env var)
            access_token: Long-lived access token (default: auto-load from secrets or env var)
            cache_ttl: Cache TTL in seconds for device states
        """
        # Load from secrets file if not provided
        if not api_url or not access_token:
            secrets = self._load_secrets()
            api_url = api_url or secrets.get("url") or os.getenv("HOME_ASSISTANT_URL", "")
            access_token = access_token or secrets.get("token") or os.getenv("HOME_ASSISTANT_TOKEN", "")

        self.api_url = (api_url or "").rstrip("/")
        self.access_token = access_token or ""
        self.cache_ttl = cache_ttl

        # State cache
        self._state_cache: dict[str, Any] = {}
        self._cache_timestamp = 0

        if not self.api_url:
            logger.warning(
                "HOME_ASSISTANT_URL not configured, behavior will not work"
            )

        if not self.access_token:
            logger.warning(
                "HOME_ASSISTANT_TOKEN not configured, behavior will not work"
            )

    def _load_secrets(self) -> dict:
        """Load Home Assistant secrets from .jetbox/secrets/home_assistant.json.

        Returns:
            Dict with 'url' and 'token' keys, or empty dict if file not found
        """
        from pathlib import Path

        secrets_file = Path.home() / ".jetbox" / "secrets" / "home_assistant.json"

        # Also check workspace-relative path
        if not secrets_file.exists():
            secrets_file = Path(".jetbox/secrets/home_assistant.json")

        if secrets_file.exists():
            try:
                with open(secrets_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load Home Assistant secrets: {e}")

        return {}

    def get_name(self) -> str:
        """Get behavior name."""
        return "home_assistant"

    def _api_request(
        self, endpoint: str, method: str = "GET", data: Optional[dict] = None
    ) -> Any:
        """Make a request to Home Assistant API.

        Args:
            endpoint: API endpoint (e.g., "/api/states")
            method: HTTP method
            data: Request body (for POST)

        Returns:
            API response (parsed JSON)
        """
        if not self.api_url or not self.access_token:
            return {
                "error": "Home Assistant not configured. Set HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN environment variables."
            }

        url = f"{self.api_url}{endpoint}"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            if data:
                request = Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers=headers,
                    method=method,
                )
            else:
                request = Request(url, headers=headers, method=method)

            with urlopen(request, timeout=10) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}

        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"Home Assistant API error: {e.code} - {error_body}")
            return {"error": f"API error: {e.code}", "details": error_body}
        except URLError as e:
            logger.error(f"Home Assistant connection error: {e.reason}")
            return {"error": f"Connection error: {e.reason}"}
        except Exception as e:
            logger.error(f"Home Assistant request failed: {e}")
            return {"error": str(e)}

    # Tool implementations (using @tool decorator)

    @tool
    def ha_list_devices(
        self, domain: Optional[str] = None, area: Optional[str] = None
    ) -> dict:
        """List all devices/entities from Home Assistant. Returns device names, states, and types.

        Args:
            domain: Optional domain filter (e.g., 'light', 'switch', 'climate', 'sensor')
            area: Optional area/room filter

        Returns:
            List of devices
        """
        states = self._api_request("/api/states")

        if "error" in states:
            return states

        # Filter by domain if specified
        if domain:
            states = [
                s for s in states if s.get("entity_id", "").startswith(f"{domain}.")
            ]

        # Filter by area if specified (requires checking attributes)
        if area:
            states = [
                s
                for s in states
                if s.get("attributes", {}).get("friendly_name", "").lower().find(
                    area.lower()
                )
                != -1
            ]

        # Format output
        devices = []
        for state in states:
            devices.append(
                {
                    "entity_id": state.get("entity_id"),
                    "name": state.get("attributes", {}).get("friendly_name"),
                    "state": state.get("state"),
                    "domain": state.get("entity_id", "").split(".")[0],
                    "attributes": state.get("attributes", {}),
                }
            )

        return {"devices": devices, "count": len(devices)}

    @tool
    def ha_get_state(self, entity_id: str) -> dict:
        """Get the current state of a specific device/entity.

        Args:
            entity_id: Entity ID (e.g., 'light.living_room', 'climate.thermostat')

        Returns:
            Entity state
        """
        state = self._api_request(f"/api/states/{entity_id}")

        if "error" in state:
            return state

        return {
            "entity_id": state.get("entity_id"),
            "name": state.get("attributes", {}).get("friendly_name"),
            "state": state.get("state"),
            "attributes": state.get("attributes", {}),
            "last_changed": state.get("last_changed"),
            "last_updated": state.get("last_updated"),
        }

    @tool
    def ha_call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[dict] = None,
    ) -> dict:
        """Call a Home Assistant service to control devices (turn on/off, set values, etc.).

        Args:
            domain: Service domain (e.g., 'light', 'switch', 'climate')
            service: Service name (e.g., 'turn_on', 'turn_off', 'set_temperature')
            entity_id: Target entity ID (e.g., 'light.living_room')
            data: Optional service data (e.g., {'brightness': 255, 'color_temp': 300})

        Returns:
            Service call result
        """
        payload = {"entity_id": entity_id}

        if data:
            payload.update(data)

        logger.info(
            f"Calling Home Assistant service: {domain}.{service} on {entity_id}"
        )

        result = self._api_request(
            f"/api/services/{domain}/{service}", method="POST", data=payload
        )

        if "error" in result:
            return result

        # Service calls return array of changed states
        return {
            "success": True,
            "service": f"{domain}.{service}",
            "entity_id": entity_id,
            "result": result,
        }

    @tool
    def ha_list_automations(self) -> dict:
        """List all configured Home Assistant automations.

        Returns:
            List of automations
        """
        states = self._api_request("/api/states")

        if "error" in states:
            return states

        # Filter to automation domain
        automations = [
            s for s in states if s.get("entity_id", "").startswith("automation.")
        ]

        # Format output
        automation_list = []
        for auto in automations:
            automation_list.append(
                {
                    "entity_id": auto.get("entity_id"),
                    "name": auto.get("attributes", {}).get("friendly_name"),
                    "state": auto.get("state"),
                    "last_triggered": auto.get("attributes", {}).get(
                        "last_triggered"
                    ),
                }
            )

        return {"automations": automation_list, "count": len(automation_list)}

    @tool
    def ha_trigger_automation(self, entity_id: str) -> dict:
        """Manually trigger a Home Assistant automation.

        Args:
            entity_id: Automation entity ID (e.g., 'automation.morning_routine')

        Returns:
            Trigger result
        """
        logger.info(f"Triggering automation: {entity_id}")

        result = self._api_request(
            "/api/services/automation/trigger",
            method="POST",
            data={"entity_id": entity_id},
        )

        if "error" in result:
            return result

        return {
            "success": True,
            "entity_id": entity_id,
            "result": result,
        }
