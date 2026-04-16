import { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Activity, AlertTriangle, Brain, Check, Copy, Mic, RefreshCw, Square, Volume2, Wifi, WifiOff } from 'lucide-react';
import { ConfigCard } from '../components/ui/ConfigCard';
import { useAuth } from '../auth/AuthContext';

type VoiceState = 'idle' | 'connecting' | 'listening' | 'speaking' | 'error';
type LogKind = 'system' | 'user' | 'assistant' | 'error';

interface LocalAiStatus {
    connected: boolean;
    status?: string;
    stt_backend?: string | null;
    tts_backend?: string | null;
    models?: {
        stt?: { backend?: string; path?: string; loaded?: boolean };
        tts?: { backend?: string; path?: string; loaded?: boolean };
        llm?: { path?: string; display?: string; loaded?: boolean };
    };
    error?: string;
}

interface SessionLogRow {
    id: string;
    kind: LogKind;
    text: string;
    ts: string;
}

const makeId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const makeWsUrl = (token: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}/api/local-ai/browser-proxy/ws?token=${encodeURIComponent(token)}`;
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const downsampleFloat32ToInt16 = (input: Float32Array, inputRate: number, targetRate: number): Int16Array => {
    if (!input.length) return new Int16Array(0);
    if (inputRate === targetRate) {
        const out = new Int16Array(input.length);
        for (let i = 0; i < input.length; i += 1) {
            const sample = clamp(input[i], -1, 1);
            out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }
        return out;
    }

    const ratio = inputRate / targetRate;
    const outputLength = Math.max(1, Math.round(input.length / ratio));
    const out = new Int16Array(outputLength);
    let pos = 0;
    for (let i = 0; i < outputLength; i += 1) {
        const nextPos = Math.min(input.length, Math.round((i + 1) * ratio));
        let sum = 0;
        let count = 0;
        for (; pos < nextPos; pos += 1) {
            sum += input[pos];
            count += 1;
        }
        const sample = clamp(count > 0 ? sum / count : input[Math.min(pos, input.length - 1)] || 0, -1, 1);
        out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return out;
};

const mulawToPcm16 = (value: number): number => {
    const mu = ~value & 0xff;
    const sign = mu & 0x80;
    const exponent = (mu >> 4) & 0x07;
    const mantissa = mu & 0x0f;
    let sample = ((mantissa << 4) + 0x08) << exponent;
    sample -= 0x84;
    return sign ? -sample : sample;
};

const decodeMulawToFloat32 = (payload: Uint8Array): Float32Array => {
    const out = new Float32Array(payload.length);
    for (let i = 0; i < payload.length; i += 1) {
        out[i] = mulawToPcm16(payload[i]) / 32768;
    }
    return out;
};

const statusTone = (state: VoiceState) => {
    switch (state) {
        case 'listening':
            return 'text-emerald-600';
        case 'speaking':
            return 'text-blue-600';
        case 'connecting':
            return 'text-amber-600';
        case 'error':
            return 'text-red-600';
        default:
            return 'text-muted-foreground';
    }
};

const BrowserVoicePage = () => {
    const { token } = useAuth();

    const [voiceState, setVoiceState] = useState<VoiceState>('idle');
    const [localStatus, setLocalStatus] = useState<LocalAiStatus | null>(null);
    const [logs, setLogs] = useState<SessionLogRow[]>([]);
    const [partialUserText, setPartialUserText] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [sampleRate, setSampleRate] = useState<number | null>(null);
    const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);
    const [refreshingStatus, setRefreshingStatus] = useState(false);
    const [sessionStartedAt, setSessionStartedAt] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);

    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);
    const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const keepAliveGainRef = useRef<GainNode | null>(null);
    const playbackEndTimeRef = useRef(0);
    const playbackTimerRef = useRef<number | null>(null);
    const finalTranscriptAtRef = useRef<number | null>(null);
    const isAgentSpeakingRef = useRef(false);
    const callIdRef = useRef(`browser-${makeId()}`);

    const connectionSummary = useMemo(() => {
        if (!localStatus?.connected) return localStatus?.error || 'local_ai_server is not reachable';
        const stt = localStatus.stt_backend || localStatus.models?.stt?.backend || 'unknown';
        const tts = localStatus.tts_backend || localStatus.models?.tts?.backend || 'unknown';
        const llm = localStatus.models?.llm?.display || localStatus.models?.llm?.path || 'unknown';
        return `STT: ${stt} · TTS: ${tts} · LLM: ${llm}`;
    }, [localStatus]);

    const pushLog = (kind: LogKind, text: string) => {
        const message = (text || '').trim();
        if (!message) return;
        setLogs(prev => [...prev.slice(-39), { id: makeId(), kind, text: message, ts: new Date().toLocaleTimeString() }]);
    };

    const clearAudioResources = async () => {
        if (playbackTimerRef.current !== null) {
            window.clearTimeout(playbackTimerRef.current);
            playbackTimerRef.current = null;
        }

        processorRef.current?.disconnect();
        sourceNodeRef.current?.disconnect();
        keepAliveGainRef.current?.disconnect();

        processorRef.current = null;
        sourceNodeRef.current = null;
        keepAliveGainRef.current = null;

        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }

        if (audioContextRef.current) {
            try {
                await audioContextRef.current.close();
            } catch {
                // ignore
            }
            audioContextRef.current = null;
        }

        playbackEndTimeRef.current = 0;
        isAgentSpeakingRef.current = false;
        finalTranscriptAtRef.current = null;
        setSampleRate(null);
    };

    const stopSession = async (reason?: string) => {
        const ws = wsRef.current;
        wsRef.current = null;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close(1000, 'client-stop');
        } else if (ws && ws.readyState === WebSocket.CONNECTING) {
            ws.close();
        }
        await clearAudioResources();
        setPartialUserText('');
        setSessionStartedAt(null);
        setVoiceState('idle');
        if (reason) pushLog('system', reason);
    };

    const schedulePlayback = (payload: ArrayBuffer) => {
        const ctx = audioContextRef.current;
        if (!ctx) return;

        const floatSamples = decodeMulawToFloat32(new Uint8Array(payload));
        const audioBuffer = ctx.createBuffer(1, floatSamples.length, 8000);
        audioBuffer.copyToChannel(floatSamples, 0);

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);

        const startAt = Math.max(ctx.currentTime + 0.02, playbackEndTimeRef.current);
        source.start(startAt);
        playbackEndTimeRef.current = startAt + audioBuffer.duration;
        isAgentSpeakingRef.current = true;
        setVoiceState('speaking');

        if (finalTranscriptAtRef.current != null) {
            setLastLatencyMs(Math.round(performance.now() - finalTranscriptAtRef.current));
            finalTranscriptAtRef.current = null;
        }

        if (playbackTimerRef.current !== null) {
            window.clearTimeout(playbackTimerRef.current);
        }
        const remainingMs = Math.max(120, Math.round((playbackEndTimeRef.current - ctx.currentTime) * 1000) + 50);
        playbackTimerRef.current = window.setTimeout(() => {
            isAgentSpeakingRef.current = false;
            setVoiceState(wsRef.current?.readyState === WebSocket.OPEN ? 'listening' : 'idle');
            playbackTimerRef.current = null;
        }, remainingMs);

        source.onended = () => {
            if (ctx.currentTime >= playbackEndTimeRef.current - 0.05) {
                isAgentSpeakingRef.current = false;
                setVoiceState(wsRef.current?.readyState === WebSocket.OPEN ? 'listening' : 'idle');
            }
        };
    };

    const refreshStatus = async () => {
        try {
            setRefreshingStatus(true);
            const res = await axios.get('/api/local-ai/status');
            setLocalStatus(res.data as LocalAiStatus);
        } catch (err: any) {
            setLocalStatus({ connected: false, error: err?.response?.data?.detail || err?.message || 'Failed to load status' });
        } finally {
            setRefreshingStatus(false);
        }
    };

    const startSession = async () => {
        if (!token) {
            setError('Missing auth token');
            setVoiceState('error');
            return;
        }

        setError(null);
        setPartialUserText('');
        setLastLatencyMs(null);
        setVoiceState('connecting');
        callIdRef.current = `browser-${makeId()}`;

        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    channelCount: 1,
                },
            });
            mediaStreamRef.current = mediaStream;

            const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
            if (!AudioContextCtor) {
                throw new Error('Browser AudioContext is not supported');
            }
            const ctx = new AudioContextCtor({ latencyHint: 'interactive' });
            audioContextRef.current = ctx;
            await ctx.resume();
            setSampleRate(ctx.sampleRate);

            const ws = new WebSocket(makeWsUrl(token));
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
                if (typeof event.data === 'string') {
                    let payload: Record<string, unknown>;
                    try {
                        payload = JSON.parse(event.data) as Record<string, unknown>;
                    } catch {
                        pushLog('error', `Invalid server message: ${event.data}`);
                        return;
                    }

                    const type = String(payload.type || '');
                    if (type === 'mode_ready') {
                        pushLog('system', 'Voice session connected');
                        setVoiceState('listening');
                        return;
                    }
                    if (type === 'proxy_error') {
                        const message = String(payload.message || 'Proxy error');
                        setError(message);
                        pushLog('error', message);
                        setVoiceState('error');
                        setSessionStartedAt(null);
                        void clearAudioResources();
                        return;
                    }
                    if (type === 'stt_result') {
                        const text = String(payload.text || '').trim();
                        const isFinal = Boolean(payload.is_final);
                        if (!text) return;
                        if (isFinal) {
                            setPartialUserText('');
                            finalTranscriptAtRef.current = performance.now();
                            pushLog('user', text);
                        } else {
                            setPartialUserText(text);
                        }
                        return;
                    }
                    if (type === 'llm_response') {
                        const text = String(payload.text || '').trim();
                        if (text) pushLog('assistant', text);
                        return;
                    }
                    if (type === 'auth_response') return;
                    if (type === 'tts_audio') return;
                    pushLog('system', JSON.stringify(payload));
                    return;
                }

                schedulePlayback(event.data);
            };

            await new Promise<void>((resolve, reject) => {
                ws.onopen = () => resolve();
                ws.onerror = () => reject(new Error('Failed to open browser voice websocket'));
                ws.onclose = () => {
                    if (wsRef.current === ws) {
                        wsRef.current = null;
                        setPartialUserText('');
                        setSessionStartedAt(null);
                        setVoiceState(prev => (prev === 'error' ? prev : 'idle'));
                        void clearAudioResources();
                    }
                };
            });

            ws.send(JSON.stringify({ type: 'set_mode', mode: 'full', call_id: callIdRef.current }));

            const source = ctx.createMediaStreamSource(mediaStream);
            const processor = ctx.createScriptProcessor(1024, 1, 1);
            const keepAliveGain = ctx.createGain();
            keepAliveGain.gain.value = 0;

            source.connect(processor);
            processor.connect(keepAliveGain);
            keepAliveGain.connect(ctx.destination);

            processor.onaudioprocess = (audioEvent: AudioProcessingEvent) => {
                if (ws.readyState !== WebSocket.OPEN) return;
                if (isAgentSpeakingRef.current) return;

                const input = audioEvent.inputBuffer.getChannelData(0);
                const pcm = downsampleFloat32ToInt16(input, ctx.sampleRate, 16000);
                if (pcm.length > 0) {
                    ws.send(pcm.buffer);
                }
            };

            sourceNodeRef.current = source;
            processorRef.current = processor;
            keepAliveGainRef.current = keepAliveGain;
            setSessionStartedAt(new Date().toLocaleTimeString());
        } catch (err: any) {
            const message = err?.message || 'Failed to start browser voice session';
            setError(message);
            pushLog('error', message);
            setVoiceState('error');
            setSessionStartedAt(null);
            await clearAudioResources();
            const ws = wsRef.current;
            wsRef.current = null;
            if (ws) {
                try {
                    ws.close();
                } catch {
                    // ignore
                }
            }
        }
    };

    useEffect(() => {
        refreshStatus();
        const interval = window.setInterval(() => {
            refreshStatus();
        }, 10000);
        return () => {
            window.clearInterval(interval);
        };
    }, []);

    useEffect(() => {
        return () => {
            void stopSession();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleCopyTranscript = () => {
        const text = logs.map(l => `[${l.kind.toUpperCase()} ${l.ts}]\n${l.text}`).join('\n\n');
        navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    return (
        <div className="space-y-6">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Browser Voice Lab</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Low-latency browser voice path for this machine: microphone -&gt; local AI websocket -&gt; TTS back to the page.
                    </p>
                </div>
                <button
                    onClick={refreshStatus}
                    className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border hover:bg-muted text-sm"
                    disabled={refreshingStatus}
                >
                    <RefreshCw className={`w-4 h-4 ${refreshingStatus ? 'animate-spin' : ''}`} />
                    Refresh status
                </button>
            </div>

            {error && (
                <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700">
                    {error}
                </div>
            )}

            <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-6">
                <ConfigCard className="space-y-5">
                    <div className="flex items-center justify-between gap-3">
                        <div>
                            <div className="text-sm font-semibold">Voice session</div>
                            <div className="text-xs text-muted-foreground">
                                Current state: <span className={`font-medium ${statusTone(voiceState)}`}>{voiceState}</span>
                                {sessionStartedAt ? ` · started ${sessionStartedAt}` : ''}
                            </div>
                        </div>
                        {voiceState === 'idle' || voiceState === 'error' ? (
                            <button
                                onClick={() => void startSession()}
                                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 text-sm disabled:opacity-50"
                                disabled={!localStatus?.connected || !token || voiceState === 'connecting'}
                            >
                                <Mic className="w-4 h-4" />
                                Start talking
                            </button>
                        ) : (
                            <button
                                onClick={() => void stopSession('Voice session stopped')}
                                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border hover:bg-muted text-sm"
                            >
                                <Square className="w-4 h-4" />
                                Stop
                            </button>
                        )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div className="rounded-lg border border-border bg-muted/20 p-3">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <Activity className="w-4 h-4" />
                                Browser audio
                            </div>
                            <div className="mt-2 text-sm font-medium">
                                {sampleRate ? `${sampleRate} Hz input` : 'Not started'}
                            </div>
                            <div className="mt-1 text-xs text-muted-foreground">
                                Mic frames are muted while AI audio is playing to avoid feedback.
                            </div>
                        </div>
                        <div className="rounded-lg border border-border bg-muted/20 p-3">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <Brain className="w-4 h-4" />
                                Local AI
                            </div>
                            <div className="mt-2 text-sm font-medium">{localStatus?.connected ? 'Connected' : 'Unavailable'}</div>
                            <div className="mt-1 text-xs text-muted-foreground truncate" title={connectionSummary}>
                                {connectionSummary}
                            </div>
                        </div>
                        <div className="rounded-lg border border-border bg-muted/20 p-3">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <Volume2 className="w-4 h-4" />
                                Last voice latency
                            </div>
                            <div className="mt-2 text-sm font-medium">{lastLatencyMs != null ? `${lastLatencyMs} ms` : 'No sample yet'}</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                                Measured from final STT event to first TTS audio arrival.
                            </div>
                        </div>
                    </div>

                    <div className="rounded-lg border border-border bg-muted/10 p-4">
                        <div className="flex items-center justify-between mb-3">
                            <div className="text-sm font-semibold">Live transcript</div>
                            {logs.length > 0 && (
                                <button
                                    onClick={handleCopyTranscript}
                                    className="inline-flex items-center gap-1.5 px-2 py-1 text-xs font-medium rounded border bg-background hover:bg-muted transition-colors"
                                    title="Copy transcript to clipboard"
                                >
                                    {copied ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3 text-muted-foreground" />}
                                    {copied ? 'Copied' : 'Copy'}
                                </button>
                            )}
                        </div>
                        <div className="space-y-3 max-h-[420px] overflow-auto pr-1">
                            {logs.length === 0 ? (
                                <div className="text-sm text-muted-foreground">No speech yet. Start the session and say a short phrase.</div>
                            ) : (
                                logs.map(row => (
                                    <div
                                        key={row.id}
                                        className={`rounded-lg border px-3 py-2 text-sm ${
                                            row.kind === 'assistant'
                                                ? 'border-blue-200 bg-blue-50 text-blue-900'
                                                : row.kind === 'user'
                                                  ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
                                                  : row.kind === 'error'
                                                    ? 'border-red-200 bg-red-50 text-red-900'
                                                    : 'border-border bg-background'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between gap-3 text-[11px] uppercase tracking-wide opacity-70">
                                            <span>{row.kind}</span>
                                            <span>{row.ts}</span>
                                        </div>
                                        <div className="mt-1 whitespace-pre-wrap break-words">{row.text}</div>
                                    </div>
                                ))
                            )}
                            {partialUserText && (
                                <div className="rounded-lg border border-dashed border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                                    <div className="text-[11px] uppercase tracking-wide opacity-70">Listening…</div>
                                    <div className="mt-1">{partialUserText}</div>
                                </div>
                            )}
                        </div>
                    </div>
                </ConfigCard>

                <div className="space-y-6">
                    <ConfigCard className="space-y-4">
                        <div className="flex items-center gap-2">
                            {localStatus?.connected ? <Wifi className="w-4 h-4 text-emerald-600" /> : <WifiOff className="w-4 h-4 text-red-600" />}
                            <div className="text-sm font-semibold">Transport baseline</div>
                        </div>
                        <div className="text-sm text-muted-foreground">
                            This page bypasses SIP/WebRTC and talks to the local voice stack over a websocket proxy. It is the cleanest way to measure whether the current hardware can keep up without conversational lag.
                        </div>
                        <div className="rounded-lg border border-border bg-muted/20 p-3 text-sm">
                            <div className="font-medium">Recommended for lowest delay</div>
                            <ul className="mt-2 space-y-2 text-muted-foreground">
                                <li>Use headphones. Speaker playback into the mic increases delay and false triggers.</li>
                                <li>Keep the browser tab focused and avoid Bluetooth audio for first tests.</li>
                                <li>Start with short phrases. Watch the latency card after each reply.</li>
                                <li>If this page is fast but phone calls lag, the bottleneck is in SIP/Asterisk transport, not the model stack.</li>
                            </ul>
                        </div>
                    </ConfigCard>

                    <ConfigCard className="space-y-3">
                        <div className="flex items-center gap-2 text-sm font-semibold">
                            <AlertTriangle className="w-4 h-4 text-amber-600" />
                            Scope
                        </div>
                        <div className="text-sm text-muted-foreground">
                            This is not a browser SIP client for Asterisk yet. It is a voice lab page for low-latency testing on the same stack before we wire browser calling into Asterisk/WebRTC.
                        </div>
                    </ConfigCard>
                </div>
            </div>

            {/* Bottom-left fixed popup for real-time visibility */}
            <div className="fixed bottom-6 left-[280px] z-[100] w-[340px] rounded-xl border border-border/50 bg-background/95 backdrop-blur-md shadow-2xl overflow-hidden animate-in slide-in-from-bottom-5">
                <div className="bg-primary/10 px-4 py-2 flex items-center justify-between border-b">
                    <div className="flex items-center gap-2">
                        <Activity className={`w-4 h-4 ${isAgentSpeakingRef.current ? 'text-blue-500 animate-pulse' : partialUserText ? 'text-emerald-500 animate-pulse' : 'text-primary'}`} />
                        <span className="text-sm font-medium tracking-tight">Voice Copilot</span>
                    </div>
                    <div className="text-[10px] uppercase font-bold tracking-wider opacity-60">
                        {voiceState}
                    </div>
                </div>
                <div className="p-4 space-y-3">
                    <div className="flex items-center justify-between text-xs text-muted-foreground border-b border-border/50 pb-2">
                        <div>Latency: <strong className="text-foreground">{lastLatencyMs != null ? `${lastLatencyMs}ms` : '---'}</strong></div>
                        <div>Audio: <strong className="text-foreground">{sampleRate ? `${sampleRate}Hz` : '---'}</strong></div>
                    </div>
                    
                    <div>
                        <div className="text-[10px] uppercase font-bold text-emerald-600 mb-1 flex items-center gap-1">
                            <Mic className="w-3 h-3" /> You
                        </div>
                        <div className="text-sm border-l-2 border-emerald-500/30 pl-2 min-h-[1.5rem] italic text-muted-foreground line-clamp-2">
                            {partialUserText || (logs.filter((l: any) => l.kind === 'user').pop()?.text) || "Listening..."}
                        </div>
                    </div>

                    <div>
                        <div className="text-[10px] uppercase font-bold text-blue-600 mb-1 flex items-center gap-1">
                            <Brain className="w-3 h-3" /> AI Model
                        </div>
                        <div className="text-sm border-l-2 border-blue-500/30 pl-2 min-h-[1.5rem] text-foreground line-clamp-3">
                            {isAgentSpeakingRef.current ? (logs.filter((l: any) => l.kind === 'assistant').pop()?.text) || "Speaking..." : (partialUserText ? "Processing..." : "Waiting...")}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BrowserVoicePage;
