/**
 * AnswerPanel — displays the generated answer from the RAG pipeline.
 */
interface Props {
  answer: string
}

export default function AnswerPanel({ answer }: Props) {
  if (!answer) return null
  return (
    <section className="panel answer-panel">
      <h3>Answer</h3>
      <p>{answer}</p>
    </section>
  )
}
