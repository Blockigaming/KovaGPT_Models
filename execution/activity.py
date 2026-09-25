"""Format only persisted stage-start events into truthful, content-free activity."""

from datetime import datetime, timezone

from execution.contracts import require


_TEXT = {
    "planning": ("planning", "Planning the response", "Organizing the next part of the response."),
    "answer": ("writing", "Preparing the response", "Producing or revising a response from the available context."),
    "critic": ("checking", "Checking the draft", "Reviewing the draft for concrete issues."),
    "verification": ("verifying", "Verifying the response", "Checking the response before presenting it."),
    "specialist": ("thinking", "Working on a specialist review", "A specialist stage is now running."),
    "disagreement-check": ("comparing", "Comparing specialist results", "Comparing completed specialist conclusions for disagreements."),
    "judge": ("checking", "Evaluating the evidence", "Assessing the recorded disagreements and their supporting evidence."),
    "debate-round-1": ("comparing", "Examining a disagreement", "Running the single conditional challenge of a recorded material disagreement."),
    "synthesis": ("writing", "Preparing the final response", "Combining completed work into the final Kova response."),
}


def activity_from_started_event(spec, event):
    """Call only on an authenticated journal replay, not client-supplied event JSON.

    No timer, percentage, tool logo, source URL or web-search claim is fabricated.
    The operation ID identifies a local stage invocation, not proof of GPU start.
    """
    require(isinstance(event, dict) and set(event) == {"job_id", "sequence", "type", "stage_id", "occurred_ms"},
            "invalid persisted execution event")
    if event["type"] != "stage_started":
        return None
    stages = {s.id: s for s in spec.stages}
    stage = stages.get(event["stage_id"])
    require(stage is not None, "event refers to an unknown stage")
    if not stage.activity:
        return None
    key = stage.id if stage.id in _TEXT else stage.id.rsplit("-", 1)[0]
    require(key in _TEXT, "unknown activity phase")
    phase, title, summary = _TEXT[key]
    return {
        "type": "activity", "event_id": f"{event['job_id']}:{event['sequence']}",
        "request_id": event["job_id"], "sequence": event["sequence"],
        "phase": phase, "title": title, "summary": summary,
        "occurred_at": datetime.fromtimestamp(event["occurred_ms"] / 1000, timezone.utc).isoformat(),
        "grounding_operation_id": f"{event['job_id']}:{stage.id}",
    }
