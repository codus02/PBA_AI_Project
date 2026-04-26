'use client';

import { use, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { saveEvaluation } from '@/lib/api';
import type { BrewStatus, DevicePayload } from '@/lib/types';
import BrewProgress from '@/components/brew/BrewProgress';
import DevicePreview from '@/components/brew/DevicePreview';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import { cn } from '@/lib/utils';

const MOCK_PAYLOAD: DevicePayload = {
  sessionId: 'preview',
  guestId: 'preview',
  cocktailId: 'preview-2',
  cocktailName: '휴고 스프리츠',
  recipe: [
    { ingredient: '프로세코', amount: 90, unit: 'ml' },
    { ingredient: '엘더플라워 코디얼', amount: 30, unit: 'ml' },
    { ingredient: '탄산수', amount: 30, unit: 'ml' },
  ],
  adjustments: [],
  timestamp: new Date().toISOString(),
  metadata: { temperature: '4°C', mixingSpeed: 'medium', glassType: 'highball' },
};

const BREW_SEQUENCE: { status: BrewStatus; progress: number; delay: number }[] = [
  { status: 'preparing', progress: 10, delay: 0 },
  { status: 'preparing', progress: 25, delay: 1500 },
  { status: 'pouring', progress: 40, delay: 3000 },
  { status: 'pouring', progress: 60, delay: 4500 },
  { status: 'mixing', progress: 75, delay: 6000 },
  { status: 'mixing', progress: 90, delay: 9000 },
  { status: 'complete', progress: 100, delay: 12000 },
];

export default function BrewPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const searchParams = useSearchParams();
  const isPreview = searchParams.get('preview') === 'true';
  const guest = useGuest(id, guestId);
  const store = usePartyStore();
  const started = useRef(false);
  const [previewStatus, setPreviewStatus] = useState<BrewStatus>('idle');
  const [previewProgress, setPreviewProgress] = useState(0);
  const previewStarted = useRef(false);
  const [previewScore, setPreviewScore] = useState(0);

  useEffect(() => {
    if (isPreview && !previewStarted.current) {
      previewStarted.current = true;
      BREW_SEQUENCE.forEach(({ status, progress, delay }) => {
        setTimeout(() => {
          setPreviewStatus(status);
          setPreviewProgress(progress);
        }, delay);
      });
    }
  }, [isPreview]);

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

  if (isPreview) {
    return (
      <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
        <div className="mb-6">
          <span className="text-xs text-zinc-500">← 파티로</span>
          <h1 className="text-xl font-bold text-zinc-100 mt-1">미리보기님의 칵테일 제조</h1>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-2xl">🌸</span>
            <span className="text-xl font-bold text-zinc-100">휴고 스프리츠</span>
          </div>
          <span className="inline-block mt-2 px-2 py-0.5 rounded text-xs bg-amber-400/10 text-amber-400 border border-amber-400/20">
            디자인 미리보기 모드
          </span>
        </div>
        <div className="flex flex-col gap-5">
          <BrewProgress status={previewStatus} progress={previewProgress} />
          <DevicePreview payload={MOCK_PAYLOAD} />
          {previewStatus === 'complete' && (
            <div className="flex flex-col gap-3">
              <Card>
                <CardHeader><CardTitle>칵테일, 맛있었나요?</CardTitle></CardHeader>
                <CardBody>
                  <div className="flex justify-center gap-3">
                    {[1, 2, 3, 4, 5].map((score) => (
                      <button
                        key={score}
                        type="button"
                        onClick={() => setPreviewScore(score)}
                        className={cn(
                          'text-2xl transition-all hover:scale-125',
                          previewScore && score <= previewScore ? 'opacity-100' : 'opacity-30 hover:opacity-70'
                        )}
                      >
                        ⭐
                      </button>
                    ))}
                  </div>
                  {previewScore > 0 && (
                    <p className="text-center text-xs text-zinc-400 mt-2">{previewScore}점 평가 완료!</p>
                  )}
                </CardBody>
              </Card>
              <Button size="lg" className="w-full" onClick={() => {}}>
                파티 대시보드로 →
              </Button>
              <Button variant="secondary" size="lg" className="w-full" onClick={() => {}}>
                📋 세션 로그 보기
              </Button>
            </div>
          )}
        </div>
      </main>
    );
  }

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

  const handleSatisfaction = async (score: number) => {
    store.setSatisfaction(id, guestId, score);
    try {
      if (guest.dbId && guest.finalRecommendationId) {
        await saveEvaluation(guest.dbId, guest.finalRecommendationId, score, true);
      }
    } catch {
      // 평가 저장 실패해도 UI 계속
    }
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="mb-6">
        <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
          ← 파티로
        </Link>
        <h1 className="text-xl font-bold text-zinc-100 mt-1">{guest.name}님의 칵테일 제조</h1>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-2xl">{finalRecommendation.imageEmoji}</span>
          <span className="text-xl font-bold text-zinc-100">{finalRecommendation.name}</span>
        </div>
      </div>

      <div className="flex flex-col gap-5">
        <BrewProgress status={brewState.status} progress={brewState.progress} />

        {brewState.devicePayload && (
          <DevicePreview payload={brewState.devicePayload} />
        )}

        {brewState.status === 'complete' && (
          <div className="flex flex-col gap-3">
            <Card>
              <CardHeader><CardTitle>칵테일, 맛있었나요?</CardTitle></CardHeader>
              <CardBody>
                <div className="flex justify-center gap-3">
                  {[1, 2, 3, 4, 5].map((score) => (
                    <button
                      key={score}
                      type="button"
                      onClick={() => handleSatisfaction(score)}
                      className={cn(
                        'text-2xl transition-all hover:scale-125',
                        guest.satisfaction && score <= guest.satisfaction ? 'opacity-100' : 'opacity-30 hover:opacity-70'
                      )}
                    >
                      ⭐
                    </button>
                  ))}
                </div>
                {guest.satisfaction && (
                  <p className="text-center text-xs text-zinc-400 mt-2">{guest.satisfaction}점 평가 완료!</p>
                )}
              </CardBody>
            </Card>
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
