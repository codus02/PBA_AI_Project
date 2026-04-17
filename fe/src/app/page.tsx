'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { usePartyStore } from '@/lib/store/partyStore';
import Button from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import Card, { CardBody } from '@/components/ui/Card';

export default function HomePage() {
  const router = useRouter();
  const createParty = usePartyStore((s) => s.createParty);
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionCode, setSessionCode] = useState('');

  const handleCreate = () => {
    if (!name.trim()) return;
    setLoading(true);
    const party = createParty(name.trim());
    setSessionCode(party.code);
    setTimeout(() => {
      router.push(`/party/${party.id}`);
    }, 800);
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-amber-400/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 left-1/4 w-64 h-64 bg-purple-400/5 rounded-full blur-3xl" />
      </div>

      <div className="w-full max-w-md relative z-10">
        <div className="text-center mb-10">
          <div className="text-6xl mb-4">🍸</div>
          <h1 className="text-4xl font-bold text-zinc-100 tracking-tight">
            Bar<span className="text-amber-400">AI</span>
          </h1>
          <p className="text-zinc-400 mt-2 text-sm leading-relaxed">
            AI가 취향을 분석해<br />파티에 딱 맞는 칵테일을 추천해드려요
          </p>
        </div>

        <Card glow>
          <CardBody className="flex flex-col gap-4">
            <div>
              <p className="text-xs text-zinc-500 uppercase tracking-widest mb-3">새 파티 시작</p>
              <Input
                placeholder="파티 이름을 입력해주세요"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                autoFocus
              />
              <p className="text-xs text-zinc-600 mt-1.5">예: 생일파티, 홈파티, 금요일 모임</p>
            </div>

            <Button
              size="lg"
              className="w-full"
              loading={loading}
              disabled={!name.trim()}
              onClick={handleCreate}
            >
              파티 시작하기
            </Button>

            {/* {sessionCode && (
              <div className="flex items-center justify-center gap-2 bg-amber-400/10 border border-amber-400/20 rounded-xl py-2.5">
                <span className="text-xs text-zinc-400">세션 코드</span>
                <span className="font-mono font-bold text-amber-400 text-lg tracking-widest">
                  {sessionCode}
                </span>
              </div>
            )} */}
          </CardBody>
        </Card>

        <div className="mt-6 grid grid-cols-3 gap-3">
          {[
            { emoji: '👥', label: '게스트 추가', desc: '여러 명 동시 진행' },
            { emoji: '🎯', label: '취향 분석', desc: '20초 취향 입력' },
            { emoji: '🍹', label: 'AI 추천', desc: '맞춤 칵테일 제조' },
          ].map((item) => (
            <div
              key={item.label}
              className="bg-zinc-900 border border-zinc-800 rounded-xl p-3 text-center"
            >
              <div className="text-2xl mb-1">{item.emoji}</div>
              <p className="text-xs font-medium text-zinc-300">{item.label}</p>
              <p className="text-xs text-zinc-600 mt-0.5">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
