"""ask_user: pause the investigation and ask for information or a document."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.agent.models import Observation, ObsStatus
from app.tools.base import ToolContext


class AskUserArgs(BaseModel):
    question: str = Field(min_length=5)
    request_upload: bool = False


class AskUserTool:
    name: ClassVar[str] = "ask_user"
    description: ClassVar[str] = (
        "Pause and ask the user for a missing fact or document that the resources do not contain. "
        "Set request_upload=true when a document (e.g. a receipt) is needed.")
    Args: ClassVar[type[BaseModel]] = AskUserArgs

    def run(self, args: AskUserArgs, ctx: ToolContext) -> Observation:
        return Observation(tool=self.name, status=ObsStatus.PAUSED, summary=args.question,
                           data={"question": args.question, "request_upload": args.request_upload})
