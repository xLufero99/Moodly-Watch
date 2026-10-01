const MAX_LENGTH = 500
const WARNING_THRESHOLD = 50

const PLACEHOLDER =
  'Ej: "Algo de ciencia ficción oscuro y lento, como Inception pero más melancólico, y con un giro fuerte"'

function MoodInput({ value, onChange, disabled = false }) {
  const remaining = MAX_LENGTH - value.length

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="mood-input" className="text-sm font-medium text-zinc-200">
        ¿Qué se te antoja ver?
      </label>
      <textarea
        id="mood-input"
        name="mood"
        rows={4}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        maxLength={MAX_LENGTH}
        disabled={disabled}
        placeholder={PLACEHOLDER}
        className="w-full resize-y rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-zinc-100 placeholder-zinc-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500/40 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
      />
      <p className="self-end text-xs text-zinc-500">
        <span className={remaining <= WARNING_THRESHOLD ? 'text-amber-400' : undefined}>
          {value.length}
        </span>
        /{MAX_LENGTH} caracteres
      </p>
    </div>
  )
}

export default MoodInput
