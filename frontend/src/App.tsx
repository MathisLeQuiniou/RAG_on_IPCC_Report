/**
 * App — root component with tab navigation between AskYourDoc and VectorStore.
 */
import { useState } from "react"
import AskYourDoc from "./components/AskYourDoc"
import VectorStore from "./components/VectorStore"
import "./App.css"

type Tab = "askYourDoc" | "vectorStore"

export default function App() {
  const [tab, setTab] = useState<Tab>("askYourDoc")

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">RAG on IPCC Report</h1>
        <nav className="tab-nav">
          <button
            className={`tab-btn ${tab === "askYourDoc" ? "tab-btn--active" : ""}`}
            onClick={() => setTab("askYourDoc")}
          >
            Ask Your Doc
          </button>
          <button
            className={`tab-btn ${tab === "vectorStore" ? "tab-btn--active" : ""}`}
            onClick={() => setTab("vectorStore")}
          >
            Vector Store
          </button>
        </nav>
      </header>

      <main className="app-main">
        {tab === "askYourDoc" ? <AskYourDoc /> : <VectorStore />}
      </main>
    </div>
  )
}
