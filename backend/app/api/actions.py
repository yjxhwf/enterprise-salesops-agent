"""Capability-token endpoints for a local demo; no identity/RBAC claims."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from backend.app.actions.schemas import ApprovalRequest, RejectRequest, ActionError


def action_router(service):
    router = APIRouter(prefix="/api/actions")

    def response(data, status=200):
        return JSONResponse(data, status_code=status, headers={"Cache-Control": "no-store"})

    def failure(error):
        code = error.code
        status = 404 if code == "ACTION_NOT_FOUND" else 403 if code == "INVALID_APPROVAL_TOKEN" else 410 if code == "APPROVAL_EXPIRED" else 409
        return response({"error": code}, status)

    @router.get("/{action_id}")
    def get_action(action_id: str):
        try:
            return response(service.state_update(action_id))
        except ActionError as error:
            return failure(error)

    async def decision(action_id, request, reject=False):
        # Manual boundary prevents FastAPI's default validation response echoing a token.
        try:
            body = (RejectRequest if reject else ApprovalRequest).model_validate(await request.json())
        except (ValidationError, ValueError, UnicodeError):
            return response({"error": "INVALID_APPROVAL_REQUEST"}, 422)
        try:
            if reject:
                service.reject(action_id, body.approval_token)
            else:
                service.approve(action_id, body.approval_token)
            return response(service.state_update(action_id))
        except ActionError as error:
            return failure(error)

    @router.post("/{action_id}/approve")
    async def approve(action_id: str, request: Request):
        return await decision(action_id, request)

    @router.post("/{action_id}/reject")
    async def reject(action_id: str, request: Request):
        return await decision(action_id, request, reject=True)

    return router
