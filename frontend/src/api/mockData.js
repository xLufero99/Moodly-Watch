const TMDB_POSTER_BASE = 'https://image.tmdb.org/t/p/w500'

const mockPoster = (path) => `${TMDB_POSTER_BASE}${path}`

// Respuesta de muestra con la forma exacta de POST /recommend.
// Algunos títulos llegan con poster_url: null (sin póster en el catálogo).
export const MOCK_RESULTS = [
  {
    id: 'tmdb-movie-496243',
    title: 'Parasite',
    media_type: 'movie',
    year: 2019,
    genres: ['Comedia', 'Drama', 'Thriller'],
    poster_url: mockPoster('/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg'),
    score: 0.97,
    explanation:
      'Tensión de clase social con humor negro y un giro que reordena todo lo anterior. Ideal si buscas algo inteligente e incómodo.',
  },
  {
    id: 'tmdb-movie-550',
    title: 'Fight Club',
    media_type: 'movie',
    year: 1999,
    genres: ['Drama', 'Thriller'],
    poster_url: mockPoster('/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg'),
    score: 0.94,
    explanation:
      'Caótico, incómodo y con una crítica social debajo de la superficie. Clásico de los que no envejecen.',
  },
  {
    id: 'tmdb-movie-155',
    title: 'The Dark Knight',
    media_type: 'movie',
    year: 2008,
    genres: ['Acción', 'Crimen', 'Drama'],
    poster_url: mockPoster('/qJ2tW6WMUDux911r6m7haRef0WH.jpg'),
    score: 0.92,
    explanation:
      'El equilibrio entre espectáculo y personaje más fino del género. Si quieres tensión bien construida, es un acierto seguro.',
  },
  {
    id: 'tmdb-movie-157336',
    title: 'Interstellar',
    media_type: 'movie',
    year: 2014,
    genres: ['Aventura', 'Drama', 'Ciencia ficción'],
    poster_url: mockPoster('/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg'),
    score: 0.9,
    explanation:
      'Ciencia ficción con peso emocional y un intercambio explícito entre amor y tiempo. Perfecto para una sesión larga.',
  },
  {
    id: 'tmdb-movie-603',
    title: 'The Matrix',
    media_type: 'movie',
    year: 1999,
    genres: ['Acción', 'Ciencia ficción'],
    poster_url: null,
    score: 0.88,
    explanation:
      'La referencia de la ciencia ficción con acción. Tiene la estética de su época, pero el ritmo sigue sosteniendo.',
  },
  {
    id: 'tmdb-tv-1396',
    title: 'Breaking Bad',
    media_type: 'tv',
    year: 2008,
    genres: ['Drama', 'Crimen'],
    poster_url: mockPoster('/ggFHVNu6YYI5L9pCfOacjizRGt.jpg'),
    score: 0.96,
    explanation:
      'Corrupción progresiva contada con una precisión casi clínica. Si te enganchó la primera temporada, aquí escala a otra cosa.',
  },
  {
    id: 'tmdb-tv-87108',
    title: 'Chernobyl',
    media_type: 'tv',
    year: 2019,
    genres: ['Drama', 'Historia', 'Misterio'],
    poster_url: mockPoster('/hlLXt2tOPT6RRnjiUmoxyG1LTFi.jpg'),
    score: 0.93,
    explanation:
      'Miniserie basada en hechos reales con ritmo de thriller investigativo. Se ve entera de un tirón.',
  },
  {
    id: 'tmdb-tv-95396',
    title: 'Severance',
    media_type: 'tv',
    year: 2022,
    genres: ['Drama', 'Ciencia ficción', 'Misterio'],
    poster_url: mockPoster('/pPHpeI2X1qEd1CS1SeyrdhZ4qn1.jpg'),
    score: 0.91,
    explanation:
      'Oficina corporativa, dualidad de identidad y paranoia constante. La premisa engancha desde el primer capítulo.',
  },
  {
    id: 'tmdb-tv-66732',
    title: 'Stranger Things',
    media_type: 'tv',
    year: 2016,
    genres: ['Ciencia ficción', 'Fantasía', 'Terror'],
    poster_url: mockPoster('/49WJfeN0moxb9IPfGn8AIqMGskD.jpg'),
    score: 0.87,
    explanation:
      'Nostalgia ochentera, tensión y un grupo de amigos. Más accessible que otra cosa, ideal para ver en compañía.',
  },
  {
    id: 'tmdb-tv-100088',
    title: 'The Last of Us',
    media_type: 'tv',
    year: 2023,
    genres: ['Drama', 'Terror'],
    poster_url: null,
    score: 0.89,
    explanation:
      'Adaptación fiel que prioriza el vínculo entre personajes sobre el ritmo de acción. Contundente capítulo a capítulo.',
  },
  {
    id: 'jikan-anime-511',
    title: 'Spirited Away',
    media_type: 'anime',
    year: 2001,
    genres: ['Animación', 'Fantasía', 'Aventura'],
    poster_url: mockPoster('/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg'),
    score: 0.95,
    explanation:
      'Fantasía cálida, un mundo lleno de secretos y un personaje que crece sin perder su curiosidad. Obra maestra de Ghibli.',
  },
  {
    id: 'jikan-anime-37205',
    title: 'Your Name.',
    media_type: 'anime',
    year: 2016,
    genres: ['Animación', 'Romance', 'Fantasía'],
    poster_url: mockPoster('/q719jXXEzOoYaps6babgKnONONX.jpg'),
    score: 0.94,
    explanation:
      'Romance interdimensional con una banda sonora que se queda. Si buscas algo emotivo y relativamente corto, funciona muchísimo.',
  },
  {
    id: 'jikan-anime-128',
    title: 'Princess Mononoke',
    media_type: 'anime',
    year: 1997,
    genres: ['Animación', 'Fantasía', 'Aventura'],
    poster_url: mockPoster('/jHWmNr7m544fJ8eItsfNk8fs2Ed.jpg'),
    score: 0.9,
    explanation:
      'Naturaleza, conflicto y una protagonista que no se doblega. Muy buena puerta de entrada al anime clásico.',
  },
  {
    id: 'jikan-anime-5114',
    title: 'Fullmetal Alchemist: Brotherhood',
    media_type: 'anime',
    year: 2009,
    genres: ['Acción', 'Aventura', 'Animación'],
    poster_url: null,
    score: 0.92,
    explanation:
      'Arco completo, ritmo impecable y uno de los mejores finales de una serie larga. La recomendación segura si te gusta la acción.',
  },
  {
    id: 'jikan-anime-16498',
    title: 'Attack on Titan',
    media_type: 'anime',
    year: 2013,
    genres: ['Acción', 'Drama', 'Animación'],
    poster_url: mockPoster('/hTP1DtLGFamjfu8WqjnuQdP1n4i.jpg'),
    score: 0.91,
    explanation:
      'Guerra, supervivencia y una mitología que se va destapando. Alterna acción con reflexiones mucho más oscuras.',
  },
]
