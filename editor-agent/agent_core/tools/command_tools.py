import asyncio
from typing import NamedTuple


class CommandResult(NamedTuple):
    stdout: str
    stderr: str
    returncode: int


async def execute_command(command: str, cwd: str | None = None) -> CommandResult:
    """Execute a shell command asynchronously."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await process.communicate()
    return CommandResult(
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        returncode=process.returncode or 0,
    )


def command_execution_status() -> str:
    return "Command execution is ENABLED and ready for automated testing/validation."
