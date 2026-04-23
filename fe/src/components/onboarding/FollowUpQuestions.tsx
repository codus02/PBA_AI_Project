'use client';

import { useState, useRef, useEffect } from 'react';
import { startDialogue, sendMessage } from '@/lib/api';

interface FollowUpQuestionsProps {
  gid: string;
  onSubmit: () => void;
}

type Message = { role: 'ai' | 'user'; text: string };

export default function FollowUpQuestions({ gid, onSubmit }: FollowUpQuestionsProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [disabled, setDisabled] = useState(true);
  const [done, setDone] = useState(false);
  const [error, setError] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!disabled && !done) inputRef.current?.focus();
  }, [disabled, done]);

  const initChat = () => {
    setError(false);
    setDisabled(true);
    startDialogue(gid)
      .then((data) => {
        setMessages([{ role: 'ai', text: data.question }]);
        setDisabled(false);
      })
      .catch(() => setError(true));
  };

  useEffect(() => {
    if (gid) initChat();
  }, [gid]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || disabled || done) return;
    setInput('');
    setDisabled(true);

    const newMsgs: Message[] = [...messages, { role: 'user', text: msg }];
    setMessages(newMsgs);

    try {
      const data = await sendMessage(gid, msg);

      if (data.should_proceed) {
        setDone(true);
        setMessages([...newMsgs, { role: 'ai', text: '취향 파악이 완료됐어요! 추천을 준비할게요 🍹' }]);
        setTimeout(() => onSubmit(), 800);
      } else {
        setMessages([...newMsgs, { role: 'ai', text: data.question ?? '' }]);
        setDisabled(false);
      }
    } catch {
      setError(true);
      setDisabled(false);
    }
  };

  if (error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, padding: 32, background: '#18181b', borderRadius: 16, border: '1px solid #3f3f46', textAlign: 'center' }}>
        <p style={{ color: '#f4f4f5', fontSize: 15, fontWeight: 600 }}>서버에 연결할 수 없어요</p>
        <p style={{ color: '#71717a', fontSize: 13 }}>백엔드 서버(localhost:8000)가 실행 중인지 확인해주세요</p>
        <button
          style={{ padding: '10px 24px', borderRadius: 24, border: 'none', background: '#fbbf24', color: '#0c0a09', fontWeight: 700, fontSize: 13, cursor: 'pointer' }}
          onClick={initChat}
        >
          다시 시도
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '60vh', background: '#18181b', borderRadius: 16, border: '1px solid #3f3f46', overflow: 'hidden' }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.role === 'ai' ? 'flex-start' : 'flex-end' }}>
            <div style={{
              maxWidth: '80%', padding: '10px 14px', borderRadius: 16, fontSize: 14, lineHeight: 1.6, wordBreak: 'break-word',
              background: msg.role === 'ai' ? '#27272a' : '#fbbf24',
              color: msg.role === 'ai' ? '#f4f4f5' : '#0c0a09',
              borderBottomLeftRadius: msg.role === 'ai' ? 4 : 16,
              borderBottomRightRadius: msg.role === 'user' ? 4 : 16,
            }}>
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div style={{ display: 'flex', gap: 8, padding: '12px', borderTop: '1px solid #27272a' }}>
        <input
          style={{ flex: 1, padding: '10px 14px', borderRadius: 24, border: '1px solid #3f3f46', background: '#09090b', color: '#f4f4f5', fontSize: 14, outline: 'none' }}
          ref={inputRef}
          placeholder={done ? '완료!' : '메시지를 입력해주세요'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={disabled || done}
        />
        <button
          style={{ padding: '10px 20px', borderRadius: 24, border: 'none', background: '#fbbf24', color: '#0c0a09', fontWeight: 700, fontSize: 13, cursor: 'pointer', opacity: disabled || done ? 0.4 : 1 }}
          onClick={handleSend}
          disabled={disabled || done}
        >
          전송
        </button>
      </div>
    </div>
  );
}
