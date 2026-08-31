/**
 * QueryBar — text input and submit button for the AskYourDoc tab.
 */
interface Props {
  onSubmit: (question: string) => void
  loading: boolean
}

export default function QueryBar({ onSubmit, loading }: Props) {
  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const form = e.currentTarget
    const input = form.elements.namedItem("question") as HTMLInputElement
    const q = input.value.trim()
    if (q) onSubmit(q)
  }

  return (
    <form className="query-bar" onSubmit={handleSubmit}>
      <input
        name="question"
        type="text"
        placeholder="Ask a question about the IPCC report..."
        disabled={loading}
        autoComplete="off"
      />
      <button type="submit" disabled={loading}>
        {loading ? "Loading…" : "Ask"}
      </button>
    </form>
  )
}
