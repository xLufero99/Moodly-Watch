import { useState } from 'react'

const MEDIA_TYPE_LABELS = {
  movie: 'Película',
  tv: 'Serie',
  anime: 'Anime',
}

function Poster({ title, posterUrl }) {
  const [hasFailed, setHasFailed] = useState(false)

  // Mismo placeholder para poster_url null y para imágenes que fallan al cargar.
  if (!posterUrl || hasFailed) {
    return (
      <div className="flex aspect-2/3 w-full flex-col items-center justify-center gap-2 bg-surface-two text-mid-gray">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className="h-10 w-10"
        >
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <circle cx="9" cy="10" r="1.5" />
          <path d="M3 17l4.5-4 4 3.5L15 13l6 5" />
        </svg>
        <span className="px-3 text-center text-xs">Sin póster</span>
      </div>
    )
  }

  return (
    <img
      src={posterUrl}
      alt={`Póster de ${title}`}
      loading="lazy"
      onError={() => setHasFailed(true)}
      className="aspect-2/3 w-full bg-surface-two object-cover"
    />
  )
}

function ResultCard({ result }) {
  const { title, media_type, year, genres, poster_url, score, explanation } = result
  const affinity = Math.round(score * 100)

  return (
    <article className="flex flex-col overflow-hidden rounded-lg border border-white/10 bg-surface-one">
      <Poster title={title} posterUrl={poster_url} />

      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <span className="shrink-0 text-xs text-mid-gray">{year}</span>
        </div>

        <div className="flex items-center justify-between gap-2">
          <p className="text-xs text-mid-gray">{MEDIA_TYPE_LABELS[media_type] ?? media_type}</p>
          <span className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white">
            {affinity}% afinidad
          </span>
        </div>

        <ul className="flex flex-wrap gap-2">
          {genres.map((genre) => (
            <li
              key={genre}
              className="rounded-full border border-white/10 bg-transparent px-3 py-1 text-xs font-medium text-white"
            >
              {genre}
            </li>
          ))}
        </ul>

        <p className="mt-1 text-sm leading-relaxed text-mid-gray">{explanation}</p>
      </div>
    </article>
  )
}

export default ResultCard
