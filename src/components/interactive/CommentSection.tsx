"use client";

/**
 * Per-section comments UI. Lists existing comments (top-level + replies)
 * and renders a textarea for new ones. Sanitised HTML comes from the
 * server; we never render raw user input.
 *
 * Every failure is shown inline: a list that can't load says so (instead of
 * looking like an empty section), and a post that fails keeps the draft in
 * the textarea and explains why. Responses go through `readApiResponse`, so
 * even a non-JSON 500 becomes a message rather than an uncaught rejection.
 */
import { useEffect, useState } from "react";

import {
  networkErrorMessage,
  readApiResponse,
  type ApiResult,
} from "@/lib/api-response";

import { jsonHeadersWithSession } from "./analyticsSession";

type CommentRow = {
  id: string;
  parentId: string | null;
  userId: string;
  createdAt: string;
  hidden: boolean;
  bodyHtml: string;
};

export function CommentSection({
  sectionSlug,
  signedIn,
  currentUserId,
}: {
  sectionSlug: string;
  signedIn: boolean;
  currentUserId: string | null;
}): JSX.Element {
  const [comments, setComments] = useState<CommentRow[]>([]);
  const [body, setBody] = useState("");
  const [parentId, setParentId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const action = "load the comments";
      let result: ApiResult<CommentRow[]>;
      try {
        const res = await fetch(`/api/sections/${sectionSlug}/comments`);
        result = await readApiResponse<CommentRow[]>(res, action);
      } catch {
        result = { ok: false, error: networkErrorMessage(action) };
      }
      if (cancelled) return;
      if (result.ok) {
        setComments(result.data);
        setLoadError(null);
      } else {
        setLoadError(result.error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sectionSlug]);

  async function handlePost() {
    if (!signedIn || !body.trim() || busy) return;
    setBusy(true);
    setError(null);
    const action = "post your comment";
    try {
      let result: ApiResult<CommentRow>;
      try {
        const res = await fetch(`/api/sections/${sectionSlug}/comments`, {
          method: "POST",
          headers: jsonHeadersWithSession(),
          body: JSON.stringify({
            body,
            ...(parentId ? { parentId } : {}),
          }),
        });
        result = await readApiResponse<CommentRow>(res, action);
      } catch {
        result = { ok: false, error: networkErrorMessage(action) };
      }
      if (result.ok) {
        const created = result.data;
        setComments((prev) => [...prev, created]);
        setBody("");
        setParentId(null);
      } else {
        // The draft stays in the textarea so nothing typed is lost.
        setError(result.error);
      }
    } finally {
      setBusy(false);
    }
  }

  const topLevel = comments.filter((c) => c.parentId === null);
  const repliesOf = (id: string) => comments.filter((c) => c.parentId === id);

  return (
    <section className="mt-12 border-t border-neutral-200 pt-8 dark:border-neutral-800">
      <h2 className="font-mono text-sm uppercase tracking-widest text-accent dark:text-indigo-300">
        Comments
      </h2>

      {loadError ? (
        <p
          role="alert"
          className="mt-4 text-sm text-red-700 dark:text-red-300"
          data-testid="comments-load-error"
        >
          {loadError}
        </p>
      ) : topLevel.length === 0 ? (
        <p className="mt-4 text-sm text-neutral-500 dark:text-neutral-400">
          Be the first to leave a comment on this section.
        </p>
      ) : (
        <ol className="mt-4 space-y-4" data-testid="comments-list">
          {topLevel.map((c) => (
            <li
              key={c.id}
              className="rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-950"
            >
              <CommentBody c={c} currentUserId={currentUserId} />
              <ol className="mt-3 space-y-2 border-l-2 border-neutral-200 pl-3 dark:border-neutral-800">
                {repliesOf(c.id).map((r) => (
                  <li key={r.id}>
                    <CommentBody c={r} currentUserId={currentUserId} />
                  </li>
                ))}
              </ol>
              {signedIn && parentId !== c.id && (
                <button
                  type="button"
                  onClick={() => setParentId(c.id)}
                  className="focus-ring mt-3 rounded text-xs text-accent hover:underline dark:text-indigo-300"
                >
                  Reply
                </button>
              )}
            </li>
          ))}
        </ol>
      )}

      {signedIn ? (
        <div className="mt-6 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
          <label className="flex flex-col gap-2 text-sm">
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              {parentId ? "Replying to a comment" : "Add a comment"}
              {parentId && (
                <button
                  type="button"
                  onClick={() => setParentId(null)}
                  className="ml-2 text-xs text-neutral-500 underline dark:text-neutral-400"
                >
                  cancel
                </button>
              )}
            </span>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              maxLength={2000}
              rows={3}
              aria-label="Comment body"
              className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 font-sans text-sm dark:border-neutral-700 dark:bg-neutral-950"
              placeholder="Markdown supported. No raw HTML."
            />
          </label>
          <div className="mt-3 flex items-center gap-3">
            <button
              type="button"
              onClick={handlePost}
              disabled={busy || !body.trim()}
              className="focus-ring rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-fg hover:opacity-90 disabled:opacity-40"
            >
              {busy ? "Posting…" : "Post"}
            </button>
            {error && (
              <p
                role="alert"
                className="text-sm text-red-700 dark:text-red-300"
                data-testid="comment-post-error"
              >
                {error}
              </p>
            )}
          </div>
        </div>
      ) : (
        <p className="mt-6 rounded border border-dashed border-neutral-300 px-4 py-3 text-xs text-neutral-500 dark:border-neutral-700 dark:text-neutral-400">
          Sign in (top-right) to leave a comment.
        </p>
      )}
    </section>
  );
}

function CommentBody({
  c,
  currentUserId,
}: {
  c: CommentRow;
  currentUserId: string | null;
}) {
  if (c.hidden) {
    return (
      <p className="text-sm italic text-neutral-500 dark:text-neutral-400">
        [hidden by an admin]
      </p>
    );
  }
  const own = currentUserId === c.userId;
  return (
    <article>
      <p className="font-mono text-[10px] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        {own ? "you" : "user"} · {new Date(c.createdAt).toLocaleString()}
      </p>
      <div
        className="prose prose-sm prose-neutral dark:prose-invert mt-1 max-w-none"
        dangerouslySetInnerHTML={{ __html: c.bodyHtml }}
      />
    </article>
  );
}
