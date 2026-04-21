'use client';

import { useState, useRef, useEffect } from 'react';
import type { FollowUpAnswer } from '@/lib/types';

const CHAT_API = 'http://localhost:5000';

interface FollowUpQuestionsProps {
  questions?: unknown[];
  onSubmit: (answers: FollowUpAnswer[]) => void;
}

type Message = { role: 'ai' | 'user'; text: string; debug?: string };

export default function FollowUpQuestions({ onSubmit }: FollowUpQuestionsProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [disabled, setDisabled] = useState(true);
  const [done, setDone] = useState(false);
  const [serverError, setServerError] = useState(false);
  const history = useRef<{ question: string; answer: string }[]>([]);
  const lastAI = useRef('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!disabled && !done) inputRef.current?.focus();
  }, [disabled, done]);

  const startChat = () => {
    setServerError(false);
    setDisabled(true);
    fetch(`${CHAT_API}/api/start`, { method: 'POST' })
      .then((r) => r.json())
      .then((data) => {
        lastAI.current = data.message;
        setMessages([{ role: 'ai', text: data.message }]);
        setDisabled(false);
      })
      .catch(() => setServerError(true));
  };

  useEffect(() => {
    startChat();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sendMessage = async () => {
    const msg = input.trim();
    if (!msg || disabled || done) return;
    setInput('');
    setDisabled(true);

    history.current.push({ question: lastAI.current, answer: msg });
    const newMsgs: Message[] = [...messages, { role: 'user', text: msg }];
    setMessages(newMsgs);

    try {
      const t0 = Date.now();
      const res = await fetch(`${CHAT_API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

      lastAI.current = data.message;
      setMessages([
        ...newMsgs,
        { role: 'ai', text: data.message, debug: `⏱ ${elapsed}s\n${data.debug ?? ''}` },
      ]);

      if (data.done) {
        setDone(true);
        const answers: FollowUpAnswer[] = history.current.map((h, i) => ({
          questionId: String(i),
          question: h.question,
          answer: h.answer,
        }));
        setTimeout(() => onSubmit(answers), 800);
      } else {
        setDisabled(false);
        inputRef.current?.focus();
      }
    } catch {
      setServerError(true);
      setDisabled(false);
      inputRef.current?.focus();
    }
  };

  if (serverError) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, padding: 32, background: '#18181b', borderRadius: 16, border: '1px solid #3f3f46', textAlign: 'center' }}>
        <p style={{ color: '#f4f4f5', fontSize: 15, fontWeight: 600 }}>서버에 연결할 수 없어요</p>
        <p style={{ color: '#71717a', fontSize: 13 }}>
          채팅 서버({CHAT_API})가 실행 중인지 확인해주세요
        </p>
        <button
          style={{ padding: '10px 24px', borderRadius: 24, border: 'none', background: '#fbbf24', color: '#0c0a09', fontWeight: 700, fontSize: 13, cursor: 'pointer' }}
          onClick={startChat}
        >
          다시 시도
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '60vh', background: '#18181b', borderRadius: 16, border: '1px solid #3f3f46', overflow: 'hidden' }}>
      {/* 메시지 영역 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.role === 'ai' ? 'flex-start' : 'flex-end', gap: 4 }}>
            <div style={{
              maxWidth: '80%', padding: '10px 14px', borderRadius: 16, fontSize: 14, lineHeight: 1.6, wordBreak: 'break-word',
              background: msg.role === 'ai' ? '#27272a' : '#fbbf24',
              color: msg.role === 'ai' ? '#f4f4f5' : '#0c0a09',
              borderBottomLeftRadius: msg.role === 'ai' ? 4 : 16,
              borderBottomRightRadius: msg.role === 'user' ? 4 : 16,
            }}>
              {msg.text}
            </div>
            {msg.debug && (
              <div style={{ fontSize: 11, color: '#71717a', background: '#09090b', border: '1px solid #27272a', borderRadius: 8, padding: '5px 10px', whiteSpace: 'pre-wrap', fontFamily: 'monospace', maxWidth: '100%' }}>
                {msg.debug}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* 입력 영역 */}
      <div style={{ display: 'flex', gap: 8, padding: '12px', borderTop: '1px solid #27272a' }}>
        <input
          style={{ flex: 1, padding: '10px 14px', borderRadius: 24, border: '1px solid #3f3f46', background: '#09090b', color: '#f4f4f5', fontSize: 14, outline: 'none' }}
          ref={inputRef}
          placeholder={done ? '완료!' : '메세지를 입력해주세요'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          disabled={disabled || done}
        />
        <button
          style={{ padding: '10px 20px', borderRadius: 24, border: 'none', background: '#fbbf24', color: '#0c0a09', fontWeight: 700, fontSize: 13, cursor: 'pointer', opacity: disabled || done ? 0.4 : 1 }}
          onClick={sendMessage}
          disabled={disabled || done}
        >
          전송
        </button>
      </div>
    </div>
  );
}
