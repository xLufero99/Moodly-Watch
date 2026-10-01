import { MOCK_RESULTS } from './mockData.js'

const USE_MOCK = true

const MOCK_DELAY_MS = 800
const MOCK_RESULT_COUNT = 6
const MOCK_ERROR_TRIGGER = 'error'

// Aquí irá el fetch real a POST /recommend cuando USE_MOCK sea false.
// Recibe el payload con la forma { text, media_types, liked_ids } y debe
// devolver { results: [...] } con los mismos campos que devuelve el mock.
async function requestRecommendations(payload) {
  const response = await fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error(`La API respondió con estado ${response.status}`)
  }

  return response.json()
}

function selectMockResults({ text, media_types }) {
  // TODO: quitar este disparador de error al conectar el backend.
  // Solo existe para poder probar el estado de error de la interfaz.
  if (String(text).toLowerCase().includes(MOCK_ERROR_TRIGGER)) {
    throw new Error('Error simulado para probar el estado de error')
  }

  const selectedTypes = new Set(media_types)

  return MOCK_RESULTS.filter((item) => selectedTypes.has(item.media_type)).slice(
    0,
    MOCK_RESULT_COUNT,
  )
}

export async function getRecommendations({ text, media_types, liked_ids = [] }) {
  const payload = { text, media_types, liked_ids }

  if (!USE_MOCK) {
    return requestRecommendations(payload)
  }

  await new Promise((resolve) => setTimeout(resolve, MOCK_DELAY_MS))

  return { results: selectMockResults(payload) }
}
