// Formatação pt-BR dos números da tela. Tudo aceita null/undefined (dado que ainda não chegou) e mostra "-".
export const num = (n) => (n == null ? '-' : n.toLocaleString('pt-BR'))
export const dec = (n, casas = 3) =>
  n == null ? '-' : n.toLocaleString('pt-BR', { minimumFractionDigits: casas, maximumFractionDigits: casas })
export const pct = (parte, total) => (total ? `${Math.round((parte / total) * 100)}%` : '-')
export const sinal = (n, casas = 3) => (n == null ? '-' : `${n >= 0 ? '+' : '−'}${dec(Math.abs(n), casas)}`)
export const dataHora = (seg) =>
  seg ? new Date(seg * 1000).toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '-'
export const duracao = (seg) => (seg == null ? '-' : seg < 60 ? `${Math.round(seg)}s` : `${Math.floor(seg / 60)}min ${Math.round(seg % 60)}s`)
