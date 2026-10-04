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
      <span className="text-sm font-medium text-white">Tipos de contenido</span>
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
                  ? 'border-accent bg-accent/15 text-white'
                  : 'border-white/10 bg-transparent text-mid-gray hover:border-white/25 hover:text-white'
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
