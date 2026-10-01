const MEDIA_TYPE_OPTIONS = [
  { value: 'movie', label: 'Películas' },
  { value: 'tv', label: 'Series' },
  { value: 'anime', label: 'Anime' },
]

function MediaTypeSelector({ selected, onChange, disabled = false }) {
  function toggle(value) {
    const isSelected = selected.includes(value)

    // Siempre debe quedar al menos un tipo activo.
    if (isSelected && selected.length === 1) {
      return
    }

    onChange(isSelected ? selected.filter((item) => item !== value) : [...selected, value])
  }

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-zinc-200">Tipos de contenido</span>
      <div className="flex flex-wrap gap-2" role="group" aria-label="Tipos de contenido">
        {MEDIA_TYPE_OPTIONS.map(({ value, label }) => {
          const isActive = selected.includes(value)

          return (
            <button
              key={value}
              type="button"
              aria-pressed={isActive}
              disabled={disabled}
              onClick={() => toggle(value)}
              className={`rounded-full border px-4 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                isActive
                  ? 'border-violet-500 bg-violet-500/15 text-violet-200'
                  : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
              }`}
            >
              {label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default MediaTypeSelector
