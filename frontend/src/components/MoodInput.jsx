const MAX_LENGTH = 500
const WARNING_THRESHOLD = 50

const PLACEHOLDER =
  'Ej: "Algo de ciencia ficción oscuro y lento, como Inception pero más melancólico, y con un giro fuerte"'

function MoodInput({ value, onChange, disabled = false }) {
  const remaining = MAX_LENGTH - value.length

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="mood-input" className="text-sm font-medium text-white">
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
        className="w-full resize-y rounded-lg border border-white/10 bg-surface-one p-3 text-white placeholder:text-steel disabled:cursor-not-allowed disabled:opacity-60"
      />
      <p className="self-end text-caption text-mid-gray">
        <span className={remaining <= WARNING_THRESHOLD ? 'text-amber-400' : undefined}>
          {value.length}
        </span>
        /{MAX_LENGTH} caracteres
      </p>
    </div>
  )
}

export default MoodInput
