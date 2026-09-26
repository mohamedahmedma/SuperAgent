import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.application.ports.repositories import NewMessage, StoredMessage
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.infra.cache import RedisCache
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork
from backend.schemas.chat import normalize_rag_trace


@dataclass(frozen=True)
class MessageToStore:
    """One message a turn adds to its conversation: the role, the text, and — on an
    answer — the trace it should be stored with; on a question, the recording it was
    spoken as.

    `key` makes storing it idempotent. A save that is retried after its connection was
    lost cannot tell whether the first attempt committed; with the key, a message that
    already landed is found instead of stored twice (RAG_FIX_PLAN item 21).
    """

    message_type: str
    content: str
    rag_trace: dict | None = None
    attachment_id: str | None = None
    key: str = field(default_factory=lambda: uuid.uuid4().hex)


class ConversationStorage:
    """Conversation history: Postgres through a unit of work, with Redis in front of it.

    Both collaborators are constructor arguments. The defaults are the process-wide unit
    of work and cache; a test passes its own, and so does the composition root.

    ## Writes only ever add

    A turn appends its messages and patches the keys of the session metadata it changed.
    The save used to take the whole conversation as the caller held it and make the
    database match, deleting and re-inserting every row whenever the two disagreed. They
    disagreed more often than "the conversation was rewritten": two turns overlapping in
    one chat, or one Redis write that failed silently and left the next turn loading a
    stale copy, and the rewrite then dropped the other turn's rows and the trace — the
    images, the tables — of every message the caller had not re-supplied. An append cannot
    lose what is already stored, whatever the caller believed the conversation was.

    ## The cache is a cache

    A write invalidates the cached conversation rather than rewriting it from the caller's
    copy, and a turn loads its history from the database, which is the authority. The
    cache serves the web app's page reads, and it may be briefly stale in the one case
    Redis cache-aside always allows — a read that started before a write and finished
    after — which costs a page a refresh, never a message its row.
    """

    # What one scroll-back fetches. A conversation is read in batches because opening a
    # year-old chat should not cost the whole of it — neither the query, nor the JSON on
    # the wire, nor the asset lookups for every image it ever showed.
    DEFAULT_PAGE_SIZE = 15
    # A ceiling on what a caller may ask for, so `?limit=` cannot be used to pull an
    # unbounded conversation into memory.
    MAX_PAGE_SIZE = 200

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory = SqlAlchemyUnitOfWork,
        cache: RedisCache | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        if cache is None:
            from backend.composition import default_services

            cache = default_services().cache
        self._cache = cache

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    @staticmethod
    def _normalize_message_records(records: list[dict]) -> list[dict]:
        normalized = []
        for record in records:
            current = dict(record)
            current["rag_trace"] = normalize_rag_trace(record.get("rag_trace"))
            normalized.append(current)
        return normalized

    # -- writes ------------------------------------------------------------------------

    def append(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[MessageToStore],
        *,
        metadata: dict | None = None,
    ) -> list[int]:
        """Add `messages` to the conversation and merge `metadata` into its session.

        Returns the row ids in order. Creates the conversation on its first message. The
        metadata is a PATCH — only the keys this turn changed — merged in the database, so
        a turn never writes back a key it did not touch. Nothing when the user is unknown:
        a conversation belongs to a `users` row, and there is none to hang it off.
        """
        now = datetime.now(UTC)
        with self._unit_of_work() as uow:
            session = uow.conversations.open_session(user_id, session_id, {})
            if session is None:
                return []
            ids = list(uow.conversations.add_messages(
                session,
                [
                    NewMessage(
                        message_type=message.message_type,
                        content=str(message.content),
                        timestamp=now,
                        rag_trace=normalize_rag_trace(message.rag_trace),
                        attachment_id=message.attachment_id or None,
                        client_key=message.key,
                    )
                    for message in messages
                ],
            ))
            uow.conversations.patch_session(session, metadata=metadata or None, updated_at=now)
            uow.commit()

        self._cache.delete(self._messages_cache_key(user_id, session_id))
        self._cache.delete(self._sessions_cache_key(user_id))
        return ids

    # -- reads -------------------------------------------------------------------------

    def load(self, user_id: str, session_id: str) -> list:
        """The whole conversation, for the agent's history. Read from the database: what a
        turn is shown must be what is stored, not what a cache held a moment ago."""
        return self._to_langchain_messages(self._read_records(user_id, session_id))

    def load_for_turn(self, user_id: str, session_id: str, *, window: int) -> tuple[list, dict]:
        """What a turn reads: the latest `window` messages, oldest first, and the metadata.

        RAG_FIX_PLAN item 19. A turn used to load EVERY message, each with its trace, and
        then read only a short tail of them (see `backend.chat.context_messages.
        history_window`). Measured on a 100-message conversation: 6.5 -> 1.3 ms p50 in the
        database alone, and the traces are most of a stored answer's size, which the
        Python side then parsed and normalised for nothing. So it asks for the window, and
        for who said what, not the traces.

        Read from the database, which is the authority. And nothing is cached: the
        whole-conversation key must never hold a slice, or the web app would page through
        a chat that looks `window` messages long. Writing the full conversation there, as
        this used to, was pure cost: the turn's own save deletes it moments later.
        """
        with self._unit_of_work() as uow:
            session = uow.conversations.find_session(user_id, session_id)
            if session is None:
                return [], {}
            lines = uow.conversations.recent_dialogue(session, limit=max(1, int(window)))
            metadata = dict(session.metadata)
        return self._to_langchain_messages(
            [{"type": line.message_type, "content": line.content} for line in lines]
        ), metadata

    def list_sessions(self, user_id: str) -> list:
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = self._cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        with self._unit_of_work() as uow:
            summaries = uow.conversations.summaries(user_id)
        result = [
            {
                "session_id": summary.session_id,
                "title": summary.metadata.get("title") or summary.session_id,
                "updated_at": summary.updated_at.isoformat(),
                "message_count": summary.message_count,
            }
            for summary in summaries
        ]
        self._cache.set_json(self._sessions_cache_key(user_id), result)
        return result

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = self._cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            normalized = self._normalize_message_records(cached)
            if normalized != cached:
                self._cache.set_json(self._messages_cache_key(user_id, session_id), normalized)
            return normalized
        return self._read_records(user_id, session_id)

    def _read_records(self, user_id: str, session_id: str) -> list[dict]:
        """Every message from the database, and the cache refreshed with them."""
        with self._unit_of_work() as uow:
            session = uow.conversations.find_session(user_id, session_id)
            if session is None:
                return []
            result = [self._record(message) for message in uow.conversations.messages(session)]
        self._cache.set_json(self._messages_cache_key(user_id, session_id), result)
        return result

    @staticmethod
    def _record(message: StoredMessage) -> dict:
        record = {
            "id": message.id,
            "type": message.message_type,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "rag_trace": normalize_rag_trace(message.rag_trace),
        }
        if message.attachment_id:
            record["attachment_id"] = message.attachment_id
        return record

    def get_session_page(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        before_id: int | None = None,
    ) -> dict:
        """One batch of a conversation, newest end first.

        Opening a chat should cost the last screenful of it, not all of it — a long
        conversation is otherwise read from Postgres in full, serialised in full, and has
        every image it ever showed resolved, to render fifteen messages. Scrolling back
        asks for the batch before the oldest message on screen, which is what `before_id`
        names.

        The cursor is the message's row id rather than an offset, so messages arriving
        while someone reads cannot shift the window and make a batch repeat or skip.
        Returned oldest-first — reading order — with `has_more` saying whether anything
        older exists, established by fetching one extra row rather than by counting.
        """
        limit = max(1, min(int(limit or self.DEFAULT_PAGE_SIZE), self.MAX_PAGE_SIZE))

        cached = self._cache.get_json(self._messages_cache_key(user_id, session_id))
        # Entries written before ids were cached carry no cursor, so they cannot be paged
        # from; the database answers instead, and the next read refreshes them.
        if cached is not None and all(item.get("id") is not None for item in cached):
            return self._page_from_records(self._normalize_message_records(cached), limit, before_id)

        with self._unit_of_work() as uow:
            session = uow.conversations.find_session(user_id, session_id)
            if session is None:
                return {"messages": [], "has_more": False}
            # One row of headroom: the newest `limit` messages, plus the single row that
            # answers "is there more" without a second COUNT query.
            rows = list(uow.conversations.latest_messages(session, limit=limit + 1, before_id=before_id))

        has_more = len(rows) > limit
        window = list(reversed(rows[:limit]))
        # Deliberately not cached: this is a slice, and writing it under the
        # whole-conversation key would leave `load` believing the chat is 15 messages
        # long — the agent would lose the rest of its history.
        return {"messages": [self._record(message) for message in window], "has_more": has_more}

    @staticmethod
    def _page_from_records(records: list[dict], limit: int, before_id: int | None) -> dict:
        """The same window, taken from the cached conversation instead of the database."""
        if before_id is not None:
            records = [item for item in records if item.get("id") is not None and item["id"] < before_id]
        window = records[-limit:] if limit < len(records) else records
        return {"messages": window, "has_more": len(records) > len(window)}

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self._unit_of_work() as uow:
            if not uow.conversations.delete_session(user_id, session_id):
                return False
            uow.commit()
        self._cache.delete(self._messages_cache_key(user_id, session_id))
        self._cache.delete(self._sessions_cache_key(user_id))
        return True


__all__ = ["ConversationStorage", "MessageToStore"]
