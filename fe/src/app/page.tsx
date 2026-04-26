'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import LoadingAnimation from '@/components/ui/LoadingAnimation';
import { usePartyStore } from '@/lib/store/partyStore';
import { createPartySession } from '@/lib/api';
import Button from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import Card, { CardBody } from '@/components/ui/Card';

export default function HomePage() {
  const router = useRouter();
  const store = usePartyStore();
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [splashing, setSplashing] = useState(true);

  if (splashing) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-950">
        <LoadingAnimation loop={false} onComplete={() => setSplashing(false)} size={160} />
      </div>
    );
  }

  const handleCreate = async () => {
    if (!name.trim()) return;
    setLoading(true);
    setError('');
    try {
      const party = store.createParty(name.trim());
      const { party_session_id } = await createPartySession(name.trim());
      store.setPartyDbId(party.id, party_session_id);
      router.push(`/party/${party.id}`);
    } catch {
      setError('서버 연결 실패. 백엔드 서버가 실행 중인지 확인해주세요.');
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-amber-400/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 left-1/4 w-64 h-64 bg-purple-400/5 rounded-full blur-3xl" />
      </div>

      <div className="w-full max-w-md relative z-10">
        <div className="text-center mb-10">
          <div className="flex justify-center mb-4">
            <Image src="/mascot.png" alt="POUR YOU 마스코트" width={96} height={96} />
          </div>
          <h1 className="text-4xl font-bold tracking-tight">
            POUR <span className="text-amber-400">YOU</span>
          </h1>
          <p className="text-zinc-400 mt-2 text-sm leading-relaxed">
            AI가 취향을 분석해<br />당신에게 딱 맞는 칵테일을 추천해드려요
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

            {error && (
              <p className="text-red-400 text-xs text-center">{error}</p>
            )}

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
            { emoji: '📱', label: '취향 분석', desc: '채팅기반 취향 분석' },
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
