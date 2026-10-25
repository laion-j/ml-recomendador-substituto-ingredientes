import Passo from './Passo.jsx'
import Arquitetura from './Arquitetura.jsx'
import { PRESETS, igual } from './params.js'
import { dec } from './formatos.js'

// Um botão da tela: rótulo + valor atual à direita + controle + frase de ajuda em português simples.
function Campo({ rotulo, valor, ajuda, children }) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-stone-800">{rotulo}</span>
        <span className="text-sm font-semibold tabular-nums text-emerald-700">{valor}</span>
      </span>
      <div className="mt-1.5">{children}</div>
      <span className="mt-1 block text-xs leading-snug text-stone-500">{ajuda}</span>
    </label>
  )
}

const Slider = ({ min, max, step = 1, valor, onChange, desabilitado }) => (
  <input type="range" min={min} max={max} step={step} value={valor} disabled={desabilitado}
    onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-emerald-600 disabled:opacity-50" />
)

const Select = ({ valor, onChange, opcoes, desabilitado }) => (
  <select value={valor} disabled={desabilitado} onChange={(e) => onChange(e.target.value)}
    className="w-full rounded-lg border border-stone-300 bg-white px-2.5 py-1.5 text-sm disabled:opacity-50">
    {opcoes.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
  </select>
)

const TAXAS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
const LOTES = [32, 64, 128, 256, 512, 1024]

export default function PassoConfig({ params, setParams, dim, nExemplos, desabilitado }) {
  const mudar = (k) => (v) => setParams((p) => ({ ...p, [k]: v }))
  const camada = (i, n) => setParams((p) => ({ ...p, camadas: p.camadas.map((x, j) => (j === i ? n : x)) }))
  const preset = PRESETS.find((p) => igual(p.params, params))

  return (
    <Passo n="2" titulo="Configure a rede"
      subtitulo="Mexa nos botões e veja o desenho da rede mudar. Nada roda ainda: o treino só começa no passo 3.">
      <div className="mb-5 flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button key={p.nome} disabled={desabilitado} onClick={() => setParams(p.params)} title={p.nota}
            className={`rounded-full border px-3 py-1.5 text-sm transition disabled:opacity-50 ${preset === p ? 'border-emerald-600 bg-emerald-50 font-medium text-emerald-800' : 'border-stone-300 hover:border-stone-400'}`}>
            {p.nome}<span className="ml-1.5 hidden text-xs text-stone-400 sm:inline">{p.nota}</span>
          </button>
        ))}
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_1fr]">
        <div className="space-y-6">
          <fieldset className="space-y-3">
            <legend className="mb-1 text-xs font-semibold uppercase tracking-wide text-stone-400">A rede</legend>
            <div>
              <div className="flex items-baseline justify-between">
                <span className="text-sm font-medium text-stone-800">Camadas ocultas</span>
                <button disabled={desabilitado || params.camadas.length >= 6}
                  onClick={() => setParams((p) => ({ ...p, camadas: [...p.camadas, 32] }))}
                  className="rounded-md border border-stone-300 px-2 py-0.5 text-xs hover:bg-stone-50 disabled:opacity-40">+ camada</button>
              </div>
              <ul className="mt-2 space-y-2">
                {params.camadas.map((n, i) => (
                  <li key={i} className="flex items-center gap-3 rounded-lg bg-stone-50 px-3 py-2">
                    <span className="w-20 shrink-0 whitespace-nowrap text-xs text-stone-500">camada {i + 1}</span>
                    <Slider min={4} max={1024} step={4} valor={n} onChange={(v) => camada(i, v)} desabilitado={desabilitado} />
                    <span className="w-14 text-right text-sm font-semibold tabular-nums">{n}</span>
                    <button disabled={desabilitado || params.camadas.length <= 1} aria-label={`remover camada ${i + 1}`}
                      onClick={() => setParams((p) => ({ ...p, camadas: p.camadas.filter((_, j) => j !== i) }))}
                      className="text-stone-400 hover:text-rose-600 disabled:opacity-30">remover</button>
                  </li>
                ))}
              </ul>
              <p className="mt-1.5 text-xs text-stone-500">
                Neurônios são os "olhos" que procuram padrões. Mais camadas e mais neurônios = mais capacidade (e mais chance de decorar).
              </p>
            </div>
            <Campo rotulo="Ativação" valor={params.ativacao}
              ajuda="A função que decide quanto cada neurônio 'acende'. ReLU zera o negativo e é o padrão; tanh e ELU são mais suaves.">
              <Select valor={params.ativacao} onChange={mudar('ativacao')} desabilitado={desabilitado}
                opcoes={[['relu', 'ReLU'], ['tanh', 'tanh'], ['elu', 'ELU']]} />
            </Campo>
            <Campo rotulo="Dropout" valor={dec(params.dropout, 2)}
              ajuda="No treino, desliga esta fração de neurônios ao acaso. Obriga a rede a não depender de poucos neurônios: reduz o decoreba.">
              <Slider min={0} max={0.8} step={0.05} valor={params.dropout} onChange={mudar('dropout')} desabilitado={desabilitado} />
            </Campo>
            <Campo rotulo="Regularização L2" valor={params.l2 || 'desligada'}
              ajuda="Pune pesos grandes. Outro freio contra o overfit: pesos pequenos dão uma rede menos 'nervosa'.">
              <Select valor={params.l2} onChange={(v) => mudar('l2')(Number(v))} desabilitado={desabilitado}
                opcoes={[[0, 'desligada'], [0.00001, '0,00001'], [0.0001, '0,0001'], [0.001, '0,001'], [0.01, '0,01']]} />
            </Campo>
          </fieldset>

          <Arquitetura camadas={params.camadas} dropout={params.dropout} dim={dim} nExemplos={nExemplos} />
        </div>

        <div className="space-y-6">
          <fieldset className="space-y-3">
            <legend className="mb-1 text-xs font-semibold uppercase tracking-wide text-stone-400">O aprendizado</legend>
            <Campo rotulo="Épocas (máximo)" valor={params.epocas}
              ajuda="Quantas vezes a rede revê todos os cartões. Poucas: não aprende. Demais: decora. A paciência abaixo pode parar antes.">
              <Slider min={1} max={200} valor={params.epocas} onChange={mudar('epocas')} desabilitado={desabilitado} />
            </Campo>
            <Campo rotulo="Paciência (early stopping)" valor={params.paciencia === 0 ? 'desligada' : `${params.paciencia} épocas`}
              ajuda="Para sozinho se a perda na validação não melhorar por tantas épocas seguidas. Desligue para ver o overfit por inteiro.">
              <Slider min={0} max={20} valor={params.paciencia} onChange={mudar('paciencia')} desabilitado={desabilitado} />
            </Campo>
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" checked={params.restaurar_melhor} disabled={desabilitado} className="mt-0.5 accent-emerald-600"
                onChange={(e) => mudar('restaurar_melhor')(e.target.checked)} />
              <span><b className="font-medium">Voltar à melhor época no final.</b>
                <span className="block text-xs text-stone-500">Guarda os pesos da época com menor perda de validação e os devolve ao terminar. Desmarcado, vale a última época (já decorada).</span></span>
            </label>
            <Campo rotulo="Otimizador" valor={params.otimizador}
              ajuda="Quem decide como ajustar os pesos a cada erro. Adam se adapta sozinho (o mais usado); SGD é o clássico e mais lento; RMSprop fica no meio.">
              <Select valor={params.otimizador} onChange={mudar('otimizador')} desabilitado={desabilitado}
                opcoes={[['adam', 'Adam'], ['rmsprop', 'RMSprop'], ['sgd', 'SGD (com momentum)']]} />
            </Campo>
            <Campo rotulo="Taxa de aprendizado" valor={params.taxa}
              ajuda="O tamanho de cada passo de ajuste. Grande demais: a rede oscila e não assenta. Pequena demais: aprende devagar.">
              <Select valor={params.taxa} onChange={(v) => mudar('taxa')(Number(v))} desabilitado={desabilitado}
                opcoes={TAXAS.map((t) => [t, String(t)])} />
            </Campo>
            <Campo rotulo="Tamanho do lote" valor={params.lote}
              ajuda="Quantos cartões a rede lê antes de ajustar os pesos. Lote pequeno: ajustes ruidosos e frequentes. Grande: estáveis e mais raros.">
              <Select valor={params.lote} onChange={(v) => mudar('lote')(Number(v))} desabilitado={desabilitado}
                opcoes={LOTES.map((t) => [t, String(t)])} />
            </Campo>
          </fieldset>

          <fieldset className="space-y-3">
            <legend className="mb-1 text-xs font-semibold uppercase tracking-wide text-stone-400">Os exemplos</legend>
            <Campo rotulo="Negativos por troca real" valor={params.neg_por_pos}
              ajuda="Para cada troca real, quantos 'vizinhos que ninguém usou' entram como contraexemplo. Muda o tamanho e o equilíbrio do treino (reveja o passo 1).">
              <Slider min={1} max={10} valor={params.neg_por_pos} onChange={mudar('neg_por_pos')} desabilitado={desabilitado} />
            </Campo>
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" checked={!!params.usar_externos} disabled={desabilitado} className="mt-0.5 accent-emerald-600"
                onChange={(e) => mudar('usar_externos')(e.target.checked)} />
              <span><b className="font-medium">Treinar também com a Recipe1MSubs.</b>
                <span className="block text-xs text-stone-500">Soma ~26 mil usos de trocas de outra base (só treino; a validação continua sendo a mesma). Nos meus testes não melhorou a nota e fez a rede decorar mais cedo.</span></span>
            </label>
            <Campo rotulo="Semente" valor={params.seed}
              ajuda="Fixa o 'acaso': pesos iniciais, sorteio dos negativos e embaralhamento. Mesma semente = mesmo resultado; mude para ver o quanto o acaso pesa.">
              <input type="number" min={0} max={999999} value={params.seed} disabled={desabilitado}
                onChange={(e) => mudar('seed')(Math.max(0, Math.min(999999, Number(e.target.value) || 0)))}
                className="w-full rounded-lg border border-stone-300 px-2.5 py-1.5 text-sm disabled:opacity-50" />
            </Campo>
          </fieldset>
        </div>
      </div>
    </Passo>
  )
}
