'use client';

import { use, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { usePartyStore, useParty } from '@/lib/store/partyStore';
import { createGuestSession, uploadSpaceImage } from '@/lib/api';
import Button from '@/components/ui/Button';
import Card, { CardBody } from '@/components/ui/Card';
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
  const [showSpaceUpload, setShowSpaceUpload] = useState(false);
  const hasSpaceAnalysis = Boolean(party?.spaceAnalysis && party?.spaceImage);
  const spaceFileRef = useRef<File | null>(null);

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

  const handleAddGuest = async () => {
    if (!guestName.trim()) return;
    const guest = addGuest(id, guestName.trim());
    setGuestName('');
    setShowAdd(false);
    try {
      if (party.dbId) {
        const { guest_session_id } = await createGuestSession(party.dbId, guestName.trim());
        store.setGuestDbId(id, guest.id, guest_session_id);
        if (spaceFileRef.current) {
          try { await uploadSpaceImage(guest_session_id, spaceFileRef.current); } catch { /* 무시 */ }
          spaceFileRef.current = null;
        }
      }
    } catch {
      // DB 저장 실패해도 UI 흐름은 계속
    }
    router.push(`/party/${id}/guest/${guest.id}/onboarding`);
  };
  const completedGuests = party.guests.filter((g) => g.step === 'complete').length;

  const handleSpaceComplete = async (analysis: SpaceAnalysis, imageUrl: string, file: File) => {
    spaceFileRef.current = file;
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
          {hasSpaceAnalysis ? (
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
                    <div className="flex flex-wrap gap-1.5">
                      {(party.spaceAnalysis?.tags || []).map((tag) => (
                        <span
                          key={tag}
                          className="rounded-full border border-amber-400/20 bg-amber-400/10 px-2 py-0.5 text-xs text-amber-300"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
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
            <Card className="border-zinc-800 bg-zinc-900/50">
              <CardBody className="flex flex-col gap-4">
                <div className="flex items-start gap-3">
                  <span className="text-2xl shrink-0">📷</span>
                  <div>
                    <p className="text-sm font-semibold text-zinc-100">파티 공간 이미지를 등록해 주세요</p>
                    <p className="text-xs text-zinc-400 mt-1">
                      공간 분위기 분석이 끝나면 그다음에 게스트를 추가할 수 있어요.
                    </p>
                  </div>
                </div>
                <Button variant="secondary" onClick={() => setShowSpaceUpload(true)}>
                  이미지 등록하기
                </Button>
              </CardBody>
            </Card>
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
