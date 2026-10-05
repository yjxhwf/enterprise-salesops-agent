"""Metadata-only adapter. Reuse the frozen input models without executing tools."""

from copy import deepcopy

from pydantic import ValidationError

from backend.app.agent.schemas import ToolSelection
from backend.app.llm.gateway import ErrorCode, GatewayError
from backend.app.tools.registry import DEFINITIONS, build_default_registry


def no_database_session():
    raise RuntimeError("Planning has no database session")


class ToolCatalog:
    def __init__(self, registry=None, *, allowed_permissions=frozenset({"READ"}), input_models=None):
        if not set(allowed_permissions) <= {"READ"}:
            raise ValueError("Task 03 permits only READ capabilities")
        registry = registry if registry is not None else build_default_registry(no_database_session)
        models = input_models if input_models is not None else {item.name: item.input_model for item in DEFINITIONS}
        self._entries = {}
        for metadata in registry.list_tools():
            if metadata.permission_level not in allowed_permissions:
                continue
            model = models.get(metadata.name)
            if model is None or model.model_json_schema() != metadata.input_schema:
                raise ValueError("Catalog metadata and local input model must agree")
            self._entries[metadata.name] = (deepcopy(metadata), model)

    def capabilities(self) -> list[dict]:
        return [{"name": name, "description": meta.description, "permission_level": meta.permission_level}
                for name, (meta, _) in self._entries.items()]

    def tool_schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": name, "description": meta.description,
                 "parameters": deepcopy(meta.input_schema)}} for name, (meta, _) in self._entries.items()]

    def validate(self, selection: ToolSelection) -> ToolSelection:
        entry = self._entries.get(selection.tool_name)
        if entry is None or entry[0].permission_level != "READ":
            raise GatewayError(ErrorCode.TOOL_SELECTION_VALIDATION_ERROR)
        try:
            arguments = entry[1].model_validate(selection.arguments)
        except ValidationError:
            raise GatewayError(ErrorCode.TOOL_SELECTION_VALIDATION_ERROR) from None
        return selection.model_copy(update={"arguments": arguments.model_dump(mode="json", exclude_unset=True)})
