"use client";

/**
 * Per-section comments UI. Lists existing comments (top-level + replies)
 * and renders a textarea for new ones. Sanitised HTML comes from the
 * server; we never render raw user input.
 */
import { useEffect, useState } from "react";

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

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const res = await fetch(`/api/sections/${sectionSlug}/comments`);
      const json = (await res.json()) as
        | { ok: true; data: CommentRow[] }
        | { ok: false; error: string };
      if (!cancelled && json.ok) setComments(json.data);
    })();
    return () => {
      cancelled = true;
    };
  }, [sectionSlug]);

  async function handlePost() {
    if (!signedIn || !body.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/sections/${sectionSlug}/comments`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          body,
          ...(parentId ? { parentId } : {}),
        }),
      });
      const json = (await res.json()) as
        | { ok: true; data: CommentRow }
        | { ok: false; error: string };
      if (json.ok) {
        setComments((prev) => [...prev, json.data]);
        setBody("");
        setParentId(null);
      } else {
        setError(json.error);
      }
    } finally {
      setBusy(false);
    }
  }

  const topLevel = comments.filter((c) => c.parentId === null);
  const repliesOf = (id: string) => comments.filter((c) => c.parentId === id);

  return (
    <section className="mt-12 border-t border-neutral-200 pt-8 dark:border-neutral-800">
      <h2 className="font-mono text-sm uppercase tracking-widest text-accent">
        Comments
      </h2>

      {topLevel.length === 0 ? (
        <p className="mt-4 text-sm text-neutral-500">
          Be the first to leave a comment on this section.
        </p>
      ) : (
        <ol className="mt-4 space-y-4">
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
                  className="focus-ring mt-3 rounded text-xs text-accent hover:underline"
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
                  className="ml-2 text-xs text-neutral-500 underline"
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
              >
                {error}
              </p>
            )}
          </div>
        </div>
      ) : (
        <p className="mt-6 rounded border border-dashed border-neutral-300 px-4 py-3 text-xs text-neutral-500 dark:border-neutral-700">
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
      <p className="text-sm italic text-neutral-500">[hidden by an admin]</p>
    );
  }
  const own = currentUserId === c.userId;
  return (
    <article>
      <p className="font-mono text-[10px] uppercase tracking-widest text-neutral-500">
        {own ? "you" : "user"} · {new Date(c.createdAt).toLocaleString()}
      </p>
      <div
        className="prose prose-sm prose-neutral dark:prose-invert mt-1 max-w-none"
        dangerouslySetInnerHTML={{ __html: c.bodyHtml }}
      />
    </article>
  );
}
