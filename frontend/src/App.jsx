import { useState } from 'react'
import { getRecommendations } from './api/client'
import MediaTypeSelector from './components/MediaTypeSelector'
import MoodInput from './components/MoodInput'
import ResultList from './components/ResultList'

const INITIAL_MEDIA_TYPES = ['movie', 'tv', 'anime']

const GENERIC_ERROR_MESSAGE =
  'No pudimos obtener recomendaciones en este momento. Inténtalo de nuevo en un momento.'

function Footer() {
  return (
    <footer className="mt-16 border-t border-zinc-800/70 py-6 text-center text-xs leading-relaxed text-zinc-600">
      Este producto usa la API de TMDB pero no está avalado ni certificado por TMDB. Datos de anime
      vía Jikan (MyAnimeList).
    </footer>
  )
}

function App() {
  const [text, setText] = useState('')
  const [mediaTypes, setMediaTypes] = useState(INITIAL_MEDIA_TYPES)
  const [results, setResults] = useState([])
  const [status, setStatus] = useState('idle')
  const [errorMessage, setErrorMessage] = useState('')

  const isLoading = status === 'loading'
  const isSubmitDisabled = text.trim().length === 0 || isLoading

  async function runSearch() {
    setStatus('loading')
    setErrorMessage('')

    try {
      const data = await getRecommendations({
        text: text.trim(),
        media_types: mediaTypes,
        liked_ids: [],
      })
      setResults(data.results)
      setStatus('success')
    } catch (error) {
      setResults([])
      setErrorMessage(error?.message || GENERIC_ERROR_MESSAGE)
      setStatus('error')
    }
  }

  function handleSubmit(event) {
    event.preventDefault()

    if (isSubmitDisabled) {
      return
    }

    runSearch()
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-10 sm:px-6">
        <header className="text-center">
          <h1 className="text-3xl font-bold text-violet-400 sm:text-4xl">Moodly Watch 🎬</h1>
          <p className="mt-2 text-zinc-400">Dime qué se te antoja ver</p>
        </header>

        <main className="mt-8 flex-1">
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            <MoodInput value={text} onChange={setText} disabled={isLoading} />

            <MediaTypeSelector
              selected={mediaTypes}
              onChange={setMediaTypes}
              disabled={isLoading}
            />

            <div>
              <button
                type="submit"
                disabled={isSubmitDisabled}
                className="w-full rounded-lg bg-violet-600 px-4 py-2.5 font-semibold text-white transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500 sm:w-auto"
              >
                {isLoading ? 'Buscando…' : 'Recomendar'}
              </button>
            </div>
          </form>

          <section className="mt-10">
            <ResultList
              status={status}
              results={results}
              errorMessage={errorMessage}
              onRetry={isLoading ? undefined : runSearch}
            />
          </section>
        </main>

        <Footer />
      </div>
    </div>
  )
}

export default App
