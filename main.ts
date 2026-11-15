// free-model-aggregator-02: one server, two client wires, and a real free model behind every answer.
//
// The entry reads its brief (MAKEFUN_PROMPT_FILE): the house's table of OpenRouter's fourteen live
// `:free` models, each with the house's quality rating, and the manifest of the trial's six prompts.
// It listens on an ephemeral port, registers the port and its catalog with the env, then answers the
// env's six prompts one at a time. Every answer is a REAL completion: the chosen free model dialled
// on the grader's gateway (OPENAI_BASE_URL, Chat Completions route) on the trial's dummy key, which
// the gateway swaps for the entrant's pushed OpenRouter key.
//
// Routing: for each prompt, the candidates are the free models meeting the prompt's requirements
// (from the brief's manifest, in arrival order, intersected with what the request body itself
// declares), ordered by the house's rating, best first. A refusal is real: on a 429 the model is
// benched for the rest of the trial and the next candidate is dialled; on a 5xx the same model is
// retried once, then the next candidate; on any other refusal the next candidate. Every refusal a
// served prompt recovered from is reported in the `x-makefun-upstream-failures` header, the only
// trail the failover column can see. The `model` field of every answer and every streamed chunk
// names the dialable id actually dialled.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or SSE. The
// gateway speaks Chat Completions only, so an Anthropic-shaped request is translated into a Chat
// Completions body and the gateway's answer translated back, field by field: system prompt,
// messages, images, tools, tool choice, max_tokens, stop sequences, JSON mode, usage, stop reason.
//
// Determinism: every choice is a function of the brief, the requests and the gateway's answers.
// No wall clock, no randomness: rendered documents carry fixed timestamps and fixed ids.

import { createServer } from 'node:http';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const CALL_TIMEOUT_MS = 15000;
const MAX_ATTEMPTS_PER_PROMPT = 4;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_TOKENS = 1024;
const GATEWAY_MAX_TOKENS = 8192;
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00.000Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: the free model table and the six prompts' manifest.
// ---------------------------------------------------------------------------

interface ModelRow {
  id: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function requirementOf(v: unknown): Requirement {
  const r = isRecord(v) ? v : {};
  return { tools: r.tools === true, vision: r.vision === true, json: r.json === true };
}

function readBrief(path: string): Brief {
  const empty: Brief = { models: [], prompts: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = docs.find((d) => d.schema === 'free-model-aggregator-02/models/v1' && Array.isArray(d.models)) ?? null;
  const models: ModelRow[] =
    modelsDoc === null || !Array.isArray(modelsDoc.models)
      ? []
      : modelsDoc.models.flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [
            {
              id: asString(v.id),
              display_name: asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.id),
              rating_milli: asNumber(v.rating_milli) ?? 0,
              context_length: asNumber(v.context_length),
              tools: v.tools === true,
              vision: v.vision === true,
              json: v.json === true,
              free: v.free === true,
            },
          ];
        });
  const promptsDoc = docs.find((d) => d.schema === 'free-model-aggregator-02/prompts/v1' && Array.isArray(d.prompts)) ?? null;
  const prompts: PromptRow[] =
    promptsDoc === null || !Array.isArray(promptsDoc.prompts)
      ? []
      : promptsDoc.prompts.flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [
            {
              id: asString(v.id),
              api: v.api === 'anthropic' ? ('anthropic' as const) : ('openai' as const),
              stream: v.stream === true,
              weight: asNumber(v.weight) ?? 1,
              requires: requirementOf(v.requires),
            },
          ];
        });
  if (modelsDoc === null) return { models, prompts, error: 'the brief holds no free model table' };
  return { models, prompts, error: null };
}

// ---------------------------------------------------------------------------
// The catalog: exactly the brief's free models under the one OpenRouter provider.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: { id: string; base_url: string; key_env: string; wire: string; models: CatalogModel[] }[];
}

const brief = readBrief(PROMPT_FILE);

/** The free models of the brief, in the brief's order, each once. */
function freeModels(): ModelRow[] {
  const seen = new Set<string>();
  const out: ModelRow[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push(model);
  }
  return out;
}

function buildCatalog(): Catalog {
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: freeModels().map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}

const catalog: Catalog = buildCatalog();

// ---------------------------------------------------------------------------
// Routing: candidates by rating, with the trial's refusal memory.
// ---------------------------------------------------------------------------

/** Models that answered 429 this trial: rate-limited, so later prompts skip them outright. */
const benched = new Set<string>();
/** Which manifest row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;

function meetsRequirements(model: ModelRow, requires: Requirement): boolean {
  if (requires.tools && !model.tools) return false;
  if (requires.vision && !model.vision) return false;
  if (requires.json && !model.json) return false;
  return true;
}

/** Requirements a request body declares on its own, whichever wire it arrived on. */
function bodyRequirements(body: Json): Requirement {
  const format = isRecord(body.response_format) ? asString(body.response_format.type) : '';
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/**
 * The requirements of the prompt now arriving: the manifest's row by arrival order, intersected
 * with the body's own declarations (a body never asks for less than its manifest row, and a model
 * that cannot carry what the body holds would fail the prompt anyway).
 */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = bodyRequirements(body);
  if (entry === undefined) return fromBody;
  return {
    tools: entry.requires.tools || fromBody.tools,
    vision: entry.requires.vision || fromBody.vision,
    json: entry.requires.json || fromBody.json,
  };
}

/** The free models that can serve a prompt, best rating first, ties broken by id: a fixed order. */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels()
    .filter((m) => !benched.has(m.id) && meetsRequirements(m, requires))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

// ---------------------------------------------------------------------------
// Request translation: an Anthropic Messages body as a Chat Completions body.
// (An OpenAI-shaped body already is one; it is only re-named and capped.)
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** Anthropic's tool_choice as OpenAI's. */
function anthropicToolChoiceToOpenAI(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'none') return 'none';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

/** An Anthropic Messages request as a Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      if (typeof content === 'string') {
        messages.push({ role: 'assistant', content });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          parts.push({ type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: source.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = anthropicToolChoiceToOpenAI(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * The Chat Completions body dialled for one prompt: an OpenAI-shaped client body re-named and
 * capped, an Anthropic-shaped one translated. The gateway refuses a call with no output cap or
 * one over its cap, so a missing cap is filled and an oversized one clamped, deterministically.
 */
function gatewayBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const out: Json = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
  const cap = asNumber(out.max_tokens) ?? asNumber(out.max_completion_tokens);
  out.max_tokens = cap === null ? DEFAULT_MAX_TOKENS : Math.min(cap, GATEWAY_MAX_TOKENS);
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Answers: the gateway's Chat Completions answer, JSON or SSE, normalised once
// to {text, tool calls, usage, stop}, then rendered in the client's wire.
// ---------------------------------------------------------------------------

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface Answer {
  text: string;
  toolCalls: ToolCall[];
  input: number;
  output: number;
  stop: string;
}

const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** One SSE frame's data payloads, in order; comment and event lines are not data. */
function sseData(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) out.push(line.slice('data:'.length).replace(/^ /, ''));
    }
  }
  return out;
}

function parseOpenAIAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    const toolCalls: ToolCall[] = (Array.isArray(message.tool_calls) ? message.tool_calls : []).flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      return [
        {
          id: typeof raw.id === 'string' ? raw.id : '',
          name: typeof raw.function.name === 'string' ? raw.function.name : '',
          arguments: typeof raw.function.arguments === 'string' ? raw.function.arguments : '',
        },
      ];
    });
    return {
      text: textOf(message.content),
      toolCalls,
      input: numOrZero(usage.prompt_tokens),
      output: numOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  const calls = new Map<number, ToolCall>();
  for (const payload of sseData(body)) {
    if (payload.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const raw of delta.tool_calls) {
        if (!isRecord(raw)) continue;
        const index = typeof raw.index === 'number' ? raw.index : 0;
        const call = calls.get(index) ?? { id: '', name: '', arguments: '' };
        if (typeof raw.id === 'string' && raw.id !== '') call.id = raw.id;
        if (isRecord(raw.function)) {
          if (typeof raw.function.name === 'string' && raw.function.name !== '') call.name = raw.function.name;
          if (typeof raw.function.arguments === 'string') call.arguments += raw.function.arguments;
        }
        calls.set(index, call);
      }
    }
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, toolCalls: [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c), input, output, stop };
}

/**
 * Fill a usage gap the provider left: the wire checks want a whole input count and an output of
 * at least 1. Only a missing count is estimated, from the texts themselves, never a present one.
 */
function withUsage(answer: Answer, requestText: string): Answer {
  let { input, output } = answer;
  if (input <= 0) input = Math.max(0, Math.ceil(requestText.length / 4));
  if (output <= 0) {
    const words = answer.text.split(/\s+/).filter((w) => w !== '').length;
    output = Math.max(1, words + answer.toolCalls.length);
  }
  return { ...answer, input, output };
}

/** The stop reason in the client's wire, in the same family as the provider's. */
function stopInWire(stop: string, wire: 'openai' | 'anthropic'): string {
  if (wire === 'openai') {
    return stop === 'length' || stop === 'tool_calls' || stop === 'function_call' || stop === 'content_filter' ? stop : 'stop';
  }
  if (stop === 'length' || stop === 'max_tokens') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal') return 'refusal';
  return 'end_turn';
}

/** One server-sent event frame: an optional event line, the JSON data line, the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into fixed-size pieces for streamed deltas: a function of the text alone. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 16) pieces.push(text.slice(at, at + 16));
  return pieces;
}

/** One answer as an OpenAI Chat Completions body, or as its SSE stream. */
function renderOpenAI(answer: Answer, model: string, stream: boolean): string {
  const stop = stopInWire(answer.stop, 'openai');
  const usage = { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output };
  if (!stream) {
    const message: Json = { role: 'assistant', content: answer.text !== '' ? answer.text : answer.toolCalls.length > 0 ? null : '' };
    if (answer.toolCalls.length > 0) {
      message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
    }
    return JSON.stringify({
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message, finish_reason: stop, logprobs: null }],
      usage,
    });
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(null, chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null)),
    );
  });
  frames.push(sseEvent(null, chunk({}, stop)));
  frames.push(sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One answer as an Anthropic Messages body, or as its SSE stream. */
function renderAnthropic(answer: Answer, model: string, stream: boolean): string {
  const stop = stopInWire(answer.stop, 'anthropic');
  const blocks: Json[] = [];
  if (answer.text !== '') blocks.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    blocks.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_agg', name: call.name, input: argumentsOf(call.arguments) });
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  if (!stream) {
    return JSON.stringify({
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: blocks,
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: answer.output },
    });
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } },
    }),
  ];
  blocks.forEach((block, index) => {
    if (block.type === 'text') {
      frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
      for (const piece of splitText(asString(block.text))) {
        frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
      }
    } else {
      const call = answer.toolCalls.find((c) => c.name === block.name) ?? { arguments: '' };
      frames.push(
        sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: { type: 'tool_use', id: block.id, name: block.name, input: {} },
        }),
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  });
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Dialling the gateway, with failover, and serving one client request.
// ---------------------------------------------------------------------------

interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/** The gateway's Chat Completions route; OPENAI_BASE_URL may or may not already end in `/v1`. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The refusal code of a gateway error body: its own `code`, else its `type`, else the status. */
function refusalCode(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code;
      if (typeof value.error.code === 'number') return String(value.error.code);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type;
    }
  } catch {
    // Not JSON: the status is the code.
  }
  return `http_${status}`;
}

interface Dialled {
  answer: Answer | null;
  model: string;
  failures: Failure[];
}

/**
 * Serve one prompt: dial its candidates best-first until one answers. A 429 benches the model for
 * the rest of the trial (its rate limit is spent); a 5xx earns the same model one retry (a lone
 * upstream hiccup should not cost the prompt its best model); any other refusal moves on. At most
 * MAX_ATTEMPTS_PER_PROMPT calls are spent on one prompt. The answer is null when every attempt
 * refused; the failures are returned either way, for the trail header.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<Dialled> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  const failures: Failure[] = [];
  const requestText = JSON.stringify(body.messages ?? '');
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    const retried = { fivexx: false };
    while (attempts < MAX_ATTEMPTS_PER_PROMPT) {
      attempts += 1;
      let res: Response;
      try {
        res = await fetch(chatCompletionsUrl(), {
          method: 'POST',
          headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}`, ...(body.stream === true ? { accept: 'text/event-stream' } : {}) },
          body: JSON.stringify(gatewayBody(clientApi, body, candidate.id)),
          signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
        });
      } catch (err) {
        failures.push({ status: 0, code: 'unreachable', model: candidate.id });
        log(`dial ${candidate.id} failed: ${(err as Error).message}`);
        break;
      }
      const text = await res.text();
      if (res.ok) {
        const answer = parseOpenAIAnswer(text, body.stream === true);
        if (answer !== null) return { answer: withUsage(answer, requestText), model: candidate.id, failures };
        failures.push({ status: res.status, code: 'unparseable_answer', model: candidate.id });
        break;
      }
      const code = refusalCode(text, res.status);
      failures.push({ status: res.status, code, model: candidate.id });
      log(`dial ${candidate.id} refused: HTTP ${res.status} ${code}`);
      if (res.status === 429) {
        benched.add(candidate.id);
        break;
      }
      if (res.status >= 500 && !retried.fivexx) {
        retried.fivexx = true;
        continue;
      }
      break;
    }
  }
  return { answer: null, model: '', failures };
}

/** The refusal trail header: at most 8 rows, in the order the calls refused. */
function trailHeader(failures: Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

// ---------------------------------------------------------------------------
// The server: the two completion routes, the model list, register and poll.
// ---------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendBody(res: ServerResponse, status: number, contentType: string, text: string, extra: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  const body =
    wire === 'anthropic'
      ? { type: 'error', error: { type, message } }
      : { error: { message, type, param: null, code: type } };
  sendBody(res, status, 'application/json', JSON.stringify(body), extra);
}

/** GET /v1/models, in the wire its auth header names: x-api-key is Anthropic, Bearer is OpenAI. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = freeModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendBody(
      res,
      200,
      'application/json',
      JSON.stringify({
        data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
        has_more: false,
        first_id: list.length > 0 ? list[0].id : null,
        last_id: list.length > 0 ? list[list.length - 1].id : null,
      }),
    );
    return;
  }
  sendBody(
    res,
    200,
    'application/json',
    JSON.stringify({ object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: 'openrouter' })) }),
  );
}

/** One completion route, on either client wire: dial, fail over, render in the client's wire. */
async function completions(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const dialled = await serve(clientApi, body);
  if (dialled.answer === null) {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'every free model refused this prompt', trailHeader(dialled.failures));
  }
  const stream = body.stream === true;
  const text = clientApi === 'openai' ? renderOpenAI(dialled.answer, dialled.model, stream) : renderAnthropic(dialled.answer, dialled.model, stream);
  sendBody(res, 200, stream ? 'text/event-stream' : 'application/json', text, trailHeader(dialled.failures));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return models(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return completions('openai', req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return completions('anthropic', req, res);
  return sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print it and stop before any
    // server starts. Wait for stdout to flush so the tail is never lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${freeModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
