'use client';

import { use } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { now } from '@/lib/utils';
import type { DevicePayload } from '@/lib/types';
import CocktailCard from '@/components/recommendation/CocktailCard';
import ExplanationPanel from '@/components/recommendation/ExplanationPanel';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import { cn } from '@/lib/utils';

export default function FinalPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const guest = useGuest(id, guestId);
  const store = usePartyStore();

  if (!guest?.finalRecommendation || !guest.feedbackAnalysis || !guest.tastingRecommendation) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400 mb-3">최종 추천이 없어요</p>
          <Link href={`/party/${id}/guest/${guestId}/tasting`} className="text-amber-400 text-sm hover:underline">
            시음 단계로 가기
          </Link>
        </div>
      </main>
    );
  }

  const handleBrew = () => {
    const payload: DevicePayload = {
      sessionId: id,
      guestId,
      cocktailId: guest.finalRecommendation!.id,
      cocktailName: guest.finalRecommendation!.name,
      recipe: guest.finalRecommendation!.recipe,
      adjustments: guest.finalRecommendation!.adjustments ?? [],
      timestamp: now(),
      metadata: {
        temperature: '4°C',
        mixingSpeed: 'medium',
        glassType: 'highball',
      },
    };
    store.updateBrewState(id, guestId, {
      status: 'idle',
      progress: 0,
      devicePayload: payload,
    });
    store.updateGuestStep(id, guestId, 'brew');
    router.push(`/party/${id}/guest/${guestId}/brew`);
  };

  const handleSatisfaction = (score: number) => {
    store.setSatisfaction(id, guestId, score);
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="mb-6">
        <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
          ← 파티로
        </Link>
        <h1 className="text-xl font-bold text-zinc-100 mt-1">{guest.name}님의 최종 추천</h1>
        <p className="text-sm text-zinc-400 mt-0.5">피드백이 반영된 맞춤 칵테일이에요</p>
      </div>

      <div className="flex flex-col gap-5">
        <CocktailCard cocktail={guest.finalRecommendation} stage="final" />

        <ExplanationPanel
          initial={guest.tastingRecommendation}
          final={guest.finalRecommendation}
          feedbackAnalysis={guest.feedbackAnalysis}
        />

        {/* 만족도 평가 */}
        <Card>
          <CardHeader>
            <CardTitle>이 추천, 마음에 드세요?</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="flex justify-center gap-3">
              {[1, 2, 3, 4, 5].map((score) => (
                <button
                  key={score}
                  type="button"
                  onClick={() => handleSatisfaction(score)}
                  className={cn(
                    'text-2xl transition-all hover:scale-125',
                    guest.satisfaction && score <= guest.satisfaction
                      ? 'opacity-100'
                      : 'opacity-30 hover:opacity-70'
                  )}
                >
                  ⭐
                </button>
              ))}
            </div>
            {guest.satisfaction && (
              <p className="text-center text-xs text-zinc-400 mt-2">
                {guest.satisfaction}점 평가 완료!
              </p>
            )}
          </CardBody>
        </Card>

        <Button size="lg" className="w-full" onClick={handleBrew}>
          🍹 이 칵테일로 제조하기 →
        </Button>
      </div>
    </main>
  );
}
