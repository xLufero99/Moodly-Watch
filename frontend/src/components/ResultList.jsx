import ResultCard from './ResultCard'

function LoadingState() {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center justify-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-10 text-zinc-400"
    >
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-violet-400" />
      <span className="text-sm">Buscando algo que te encaje…</span>
    </div>
  )
}

function EmptyState({ hasSearched }) {
  return (
    <div className="rounded-xl border border-dashed border-zinc-800 px-4 py-10 text-center text-zinc-500">
      <p className="text-sm">
        {hasSearched
          ? 'No encontramos nada con esos filtros. Prueba con otra combinación de tipos.'
          : 'Cuéntanos qué te apetece y te proponemos qué ver.'}
      </p>
    </div>
  )
}

function ErrorState({ message, onRetry }) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-red-900/60 bg-red-950/30 px-4 py-6 text-center"
    >
      <p className="text-sm font-medium text-red-200">Algo salió mal</p>
      <p className="mt-1 text-sm text-red-200/80">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-lg border border-red-800 px-4 py-1.5 text-sm font-medium text-red-100 hover:bg-red-900/40"
        >
          Reintentar
        </button>
      )}
    </div>
  )
}

function ResultList({ status, results, errorMessage, onRetry }) {
  if (status === 'loading') {
    return <LoadingState />
  }

  if (status === 'error') {
    return <ErrorState message={errorMessage} onRetry={onRetry} />
  }

  if (status === 'idle' || results.length === 0) {
    return <EmptyState hasSearched={status === 'success'} />
  }

  return (
    <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {results.map((result) => (
        <li key={result.id}>
          <ResultCard result={result} />
        </li>
      ))}
    </ul>
  )
}

export { LoadingState, EmptyState, ErrorState }
export default ResultList
