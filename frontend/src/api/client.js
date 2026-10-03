import { MOCK_RESULTS } from './mockData.js'

const USE_MOCK = false

// Origen del backend. Sin VITE_API_URL queda la ruta relativa, que es lo que el
// proxy de Vite resuelve en local; en Cloudflare Pages se inyecta en el build con
// la URL del despliegue de Railway, porque ahí no hay proxy.
const API_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

const MOCK_DELAY_MS = 800
const MOCK_RESULT_COUNT = 6
const MOCK_ERROR_TRIGGER = 'error'

const INVALID_TEXT_MESSAGE = 'Revisa el texto que escribiste'
const CONNECTION_ERROR_MESSAGE = 'No pudimos conectar con el servidor'

// Petición real al backend. Recibe el payload con la forma
// { text, media_types, liked_ids } y devuelve { results: [...] } con los mismos
// campos que devuelve el mock.
async function requestRecommendations(payload) {
  try {
    const response = await fetch(`${API_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (response.status === 422) {
      throw new Error(INVALID_TEXT_MESSAGE)
    }

    if (!response.ok) {
      throw new Error(CONNECTION_ERROR_MESSAGE)
    }

    const data = await response.json()

    return { results: data.results }
  } catch (error) {
    if (error instanceof TypeError) {
      // Fallo de red: el servidor no respondió o el proxy de Vite no lo alcanzó.
      throw new Error(CONNECTION_ERROR_MESSAGE, { cause: error })
    }

    throw error
  }
}

function selectMockResults({ text, media_types }) {
  // Disparador de error para poder probar el estado de error de la interfaz.
  // Se mantiene a propósito: solo se ejecuta dentro de la rama mock, que ya no
  // es el camino por defecto desde que USE_MOCK es false.
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
