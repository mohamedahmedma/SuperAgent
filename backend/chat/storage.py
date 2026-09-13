from datetime import UTC, datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.application.ports.repositories import NewMessage, StoredMessage
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.infra.cache import RedisCache
from backend.infra.cache import cache as shared_cache
from backend.infra.unit_of_work import SqlAlchemyUnitOfWork
from backend.schemas.chat import normalize_rag_trace


class ConversationStorage:
    """Conversation history: Postgres through a unit of work, with Redis in front of it.

    Both collaborators are constructor arguments. The defaults are the process-wide unit
    of work and cache, which is what the module-level `storage` below uses; a test passes
    its own, and so will the composition root once the singletons are gone.
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
        self._cache = cache if cache is not None else shared_cache

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

    @staticmethod
    def _continues(stored: list, messages: list) -> bool:
        """Whether what is stored is still a prefix of what is being saved.

        A chat only ever grows at the end, so the ordinary case is that every stored row
        matches the message at its position and the save has a tail to append. Roles are
        compared, not content: what is stored has been through the engine's sanitiser —
        NFC normalisation, invisible characters, NUL — so it does not match the in-memory
        string for the Arabic and emoji this corpus is full of, and comparing it would
        force a needless rewrite on exactly those conversations.
        """
        if len(stored) > len(messages):
            return False
        return all(row.message_type == messages[index].type for index, row in enumerate(stored))

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        metadata: dict = None,
        extra_message_data: list = None,
    ):
        now = datetime.now(UTC)
        with self._unit_of_work() as uow:
            conversations = uow.conversations
            session = conversations.open_session(user_id, session_id, metadata or {})
            if session is None:
                return
            merged = {**session.metadata, **metadata} if metadata is not None else None

            # A turn adds one or two messages to a conversation that is otherwise
            # unchanged, so it is written as an append: the rows already stored stay
            # where they are, and only the tail is inserted.
            #
            # Deleting and re-inserting every message instead made each turn cost a
            # rewrite of the whole conversation, reset every timestamp to now, and — the
            # reason this changed — dropped the rag_trace of every message the caller did
            # not re-supply. `assets` lives on the trace, so an answer's images survived
            # only until the next message was saved.
            #
            # Heads rather than whole messages: the bodies are not needed to decide any of
            # this, and re-reading them on every save is a cost that grows with the
            # conversation.
            stored = list(conversations.message_heads(session))
            if not self._continues(stored, messages):
                # The conversation was rewritten rather than continued. Rare enough to
                # be worth handling simply: replace the lot.
                conversations.delete_messages(session)
                stored = []

            serialized = []
            appended_records = []
            appended = []
            for idx, msg in enumerate(messages):
                supplied = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    supplied = normalize_rag_trace(extra.get("rag_trace"))

                record = {"type": msg.type, "content": str(msg.content), "rag_trace": None, "id": None}
                if idx < len(stored):
                    head = stored[idx]
                    record["id"] = head.id
                    record["timestamp"] = head.timestamp.isoformat()
                    record["rag_trace"] = (
                        supplied if supplied is not None else normalize_rag_trace(head.rag_trace)
                    )
                    # Only when this save actually brought one: a turn supplies a trace
                    # for the answer it just wrote, and nothing for the history behind it.
                    if supplied is not None:
                        conversations.replace_trace(head.id, supplied)
                else:
                    record["timestamp"] = now.isoformat()
                    record["rag_trace"] = supplied
                    appended.append(
                        NewMessage(
                            message_type=msg.type,
                            content=str(msg.content),
                            timestamp=now,
                            rag_trace=supplied,
                        )
                    )
                    appended_records.append(record)
                serialized.append(record)

            # Before the commit, so the cached records carry the same row ids the
            # paginated read uses as its cursor. A cached page without them could not be
            # scrolled back from.
            for record, message_id in zip(appended_records, conversations.add_messages(session, appended)):
                record["id"] = message_id
            conversations.update_session(session, metadata=merged, updated_at=now)
            uow.commit()

        self._cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
        self._cache.delete(self._sessions_cache_key(user_id))

    def load(self, user_id: str, session_id: str) -> list:
        cached = self._cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        self._cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def load_with_meta(self, user_id: str, session_id: str) -> tuple[list, dict]:
        """Load conversation messages and session metadata (title, persistent note, etc.)."""
        messages = self.load(user_id, session_id)
        with self._unit_of_work() as uow:
            session = uow.conversations.find_session(user_id, session_id)
        return messages, dict(session.metadata) if session is not None else {}

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

        with self._unit_of_work() as uow:
            session = uow.conversations.find_session(user_id, session_id)
            if session is None:
                return []
            result = [self._record(message) for message in uow.conversations.messages(session)]
        self._cache.set_json(self._messages_cache_key(user_id, session_id), result)
        return result

    @staticmethod
    def _record(message: StoredMessage) -> dict:
        return {
            "id": message.id,
            "type": message.message_type,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "rag_trace": normalize_rag_trace(message.rag_trace),
        }

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
        # from; the database answers instead, and the next save refreshes them.
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

