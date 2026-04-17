'use client';

import { use, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { usePartyStore, useParty } from '@/lib/store/partyStore';
import Button from '@/components/ui/Button';
import Card, { CardBody } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import GuestCard from '@/components/party/GuestCard';
import { Input } from '@/components/ui/Input';
import SpaceUpload from '@/components/onboarding/SpaceUpload';
import type { SpaceAnalysis } from '@/lib/types';

export default function PartyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const party = useParty(id);
  const addGuest = usePartyStore((s) => s.addGuest);
  const store = usePartyStore();
  const [guestName, setGuestName] = useState('');
  const [showAdd, setShowAdd] = useState(false);

  if (!party) {
    return (
      <main className="min-h-screen flex flex-col items-center justify-center px-4">
        <div className="text-center">
          <div className="text-4xl mb-3">😕</div>
          <p className="text-zinc-400">파티를 찾을 수 없어요</p>
          <Link href="/" className="mt-4 inline-block text-amber-400 text-sm hover:underline">
            홈으로 돌아가기
          </Link>
        </div>
      </main>
    );
  }

  const handleAddGuest = () => {
    if (!guestName.trim()) return;
    const guest = addGuest(id, guestName.trim());
    setGuestName('');
    setShowAdd(false);
    router.push(`/party/${id}/guest/${guest.id}/onboarding`);
  };

  const completedGuests = party.guests.filter((g) => g.step === 'complete').length;
  const [showSpaceUpload, setShowSpaceUpload] = useState(false);

  const handleSpaceComplete = (analysis: SpaceAnalysis, imageUrl: string) => {
    store.setSpaceAnalysis(id, analysis);
    store.setSpaceImage(id, imageUrl);
    setShowSpaceUpload(false);
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      {/* 헤더 */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <Link href="/" className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
            ← 홈
          </Link>
          <h1 className="text-2xl font-bold text-zinc-100 mt-1">{party.name}</h1>
          {/* <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-zinc-500">세션 코드</span>
            <span className="font-mono text-amber-400 font-bold tracking-widest text-sm">{party.code}</span>
          </div> */}
        </div>
        <Link href={`/party/${id}/logs`}>
        
          <Button variant="ghost" size="sm">
            📋 로그
          </Button>
        </Link>
      </div>

      {/* 통계 */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <Card className="text-center py-4">
          <p className="text-2xl font-bold text-zinc-100">{party.guests.length}</p>
          <p className="text-xs text-zinc-500 mt-0.5">게스트</p>
        </Card>
        <Card className="text-center py-4">
          <p className="text-2xl font-bold text-amber-400">
            {party.guests.filter((g) => g.tastingRecommendation).length}
          </p>
          <p className="text-xs text-zinc-500 mt-0.5">추천 완료</p>
        </Card>
        <Card className="text-center py-4">
          <p className="text-2xl font-bold text-emerald-400">{completedGuests}</p>
          <p className="text-xs text-zinc-500 mt-0.5">제조 완료</p>
        </Card>
      </div>

      {/* 공간 설정 */}
      {showSpaceUpload ? (
        <div className="mb-6">
          <SpaceUpload
            onComplete={handleSpaceComplete}
            onSkip={() => setShowSpaceUpload(false)}
          />
        </div>
      ) : (
        <div className="mb-6">
          {party.spaceAnalysis ? (
            <Card className="border-amber-400/20">
              <CardBody>
                <div className="flex items-center gap-3">
                  {party.spaceImage && (
                    <img
                      src={party.spaceImage}
                      alt="파티 공간"
                      className="w-14 h-14 rounded-xl object-cover shrink-0"
                    />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-zinc-500 mb-0.5">파티 공간</p>
                    <p className="text-sm font-semibold text-zinc-100">{party.spaceAnalysis.style}</p>
                    <p className="text-xs text-zinc-400">{party.spaceAnalysis.mood} · {party.spaceAnalysis.colors.join(', ')}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowSpaceUpload(true)}
                    className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors shrink-0"
                  >
                    변경
                  </button>
                </div>
              </CardBody>
            </Card>
          ) : (
            <button
              type="button"
              onClick={() => setShowSpaceUpload(true)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-dashed border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/50 transition-all text-left"
            >
              <span className="text-2xl">📷</span>
              <div>
                <p className="text-sm font-medium text-zinc-300">파티 공간 이미지 추가</p>
                <p className="text-xs text-zinc-500">공간 분위기를 분석해 더 정확한 추천을 해드려요</p>
              </div>
            </button>
          )}
        </div>
      )}

      {/* 게스트 목록 */}
      <div className="flex flex-col gap-3 mb-4">
        {party.guests.length === 0 ? (
          <Card>
            <CardBody className="text-center py-8">
              <div className="text-4xl mb-3">👋</div>
              <p className="text-zinc-400 text-sm">아직 게스트가 없어요</p>
              <p className="text-zinc-600 text-xs mt-1">첫 번째 게스트를 추가해보세요</p>
            </CardBody>
          </Card>
        ) : (
          party.guests.map((guest) => (
            <GuestCard key={guest.id} guest={guest} partyId={id} />
          ))
        )}
      </div>

      {/* 게스트 추가 */}
      {showAdd ? (
        <Card glow>
          <CardBody className="flex flex-col gap-3">
            <p className="text-sm font-medium text-zinc-300">게스트 이름 입력</p>
            <Input
              placeholder="이름을 입력해주세요"
              value={guestName}
              onChange={(e) => setGuestName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddGuest()}
              autoFocus
            />
            <div className="flex gap-2">
              <Button variant="secondary" className="flex-1" onClick={() => setShowAdd(false)}>
                취소
              </Button>
              <Button className="flex-1" disabled={!guestName.trim()} onClick={handleAddGuest}>
                추가하고 시작 →
              </Button>
            </div>
          </CardBody>
        </Card>
      ) : (
        <Button
          size="lg"
          className="w-full"
          onClick={() => setShowAdd(true)}
        >
          + 게스트 추가
        </Button>
      )}
    </main>
  );
}
