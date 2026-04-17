'use client';

import { use, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import type { BrewStatus } from '@/lib/types';
import BrewProgress from '@/components/brew/BrewProgress';
import DevicePreview from '@/components/brew/DevicePreview';
import Button from '@/components/ui/Button';

const BREW_SEQUENCE: { status: BrewStatus; progress: number; delay: number }[] = [
  { status: 'preparing', progress: 10, delay: 0 },
  { status: 'preparing', progress: 25, delay: 1000 },
  { status: 'pouring', progress: 40, delay: 2500 },
  { status: 'pouring', progress: 60, delay: 4000 },
  { status: 'mixing', progress: 75, delay: 5500 },
  { status: 'mixing', progress: 90, delay: 7000 },
  { status: 'complete', progress: 100, delay: 8500 },
];

export default function BrewPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const guest = useGuest(id, guestId);
  const store = usePartyStore();
  const started = useRef(false);

  useEffect(() => {
    if (!guest || started.current) return;
    const brewState = guest.brewState;
    if (!brewState || brewState.status !== 'idle') return;

    started.current = true;

    BREW_SEQUENCE.forEach(({ status, progress, delay }) => {
      setTimeout(() => {
        store.updateBrewState(id, guestId, { status, progress });
      }, delay);
    });
  }, [guest, id, guestId, store]);

  if (!guest?.finalRecommendation || !guest.brewState) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400 mb-3">제조 정보가 없어요</p>
          <Link href={`/party/${id}/guest/${guestId}/final`} className="text-amber-400 text-sm hover:underline">
            최종 추천으로 가기
          </Link>
        </div>
      </main>
    );
  }

  const { brewState, finalRecommendation } = guest;

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="mb-6">
        <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
          ← 파티로
        </Link>
        <h1 className="text-xl font-bold text-zinc-100 mt-1">{guest.name}님의 칵테일 제조</h1>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-2xl">{finalRecommendation.imageEmoji}</span>
          <span className="text-zinc-300 font-medium">{finalRecommendation.name}</span>
        </div>
      </div>

      <div className="flex flex-col gap-5">
        <BrewProgress status={brewState.status} progress={brewState.progress} />

        {brewState.devicePayload && (
          <DevicePreview payload={brewState.devicePayload} />
        )}

        {brewState.status === 'complete' && (
          <div className="flex flex-col gap-3">
            <Link href={`/party/${id}`}>
              <Button size="lg" className="w-full">
                파티 대시보드로 →
              </Button>
            </Link>
            <Link href={`/party/${id}/logs`}>
              <Button variant="secondary" size="lg" className="w-full">
                📋 세션 로그 보기
              </Button>
            </Link>
          </div>
        )}
      </div>
    </main>
  );
}
