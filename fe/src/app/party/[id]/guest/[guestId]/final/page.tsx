'use client';

import { use } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { now } from '@/lib/utils';
import type { CocktailRecommendation, DevicePayload } from '@/lib/types';
import CocktailCard from '@/components/recommendation/CocktailCard';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';

const MOCK_TASTING: CocktailRecommendation = {
  id: 'preview-1',
  name: '아페롤 스프리츠',
  imageEmoji: '🍊',
  reason: '달콤하고 상큼한 맛, 가벼운 알코올을 선호하셔서 추천드렸어요.',
  tags: ['상큼한', '가벼운'],
  recipe: [],
};

const MOCK_FINAL: CocktailRecommendation = {
  id: 'preview-2',
  name: '휴고 스프리츠',
  description: '엘더플라워와 민트의 은은한 향이 매력적인 이탈리안 스프리츠',
  reason:
    '"좀 더 향이 강했으면 좋겠어요"라는 피드백을 반영해, 엘더플라워 코디얼을 추가하고 민트를 더한 휴고 스프리츠로 조정했어요.',
  imageEmoji: '🌸',
  tags: ['꽃향', '상큼한', '가벼운'],
  recipe: [
    { ingredient: '프로세코', amount: 90, unit: 'ml' },
    { ingredient: '엘더플라워 코디얼', amount: 30, unit: 'ml' },
    { ingredient: '탄산수', amount: 30, unit: 'ml' },
    { ingredient: '민트', amount: 3, unit: '잎' },
    { ingredient: '라임', amount: 0.5, unit: '개' },
  ],
  adjustments: [
    { ingredient: '엘더플라워 코디얼', change: '+10ml 추가', reason: '향 강도 증가 요청 반영' },
    { ingredient: '민트', change: '신규 추가', reason: '청량감과 허브향 보강' },
  ],
};

export default function FinalPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const searchParams = useSearchParams();
  const isPreview = searchParams.get('preview') === 'true';
  const guest = useGuest(id, guestId);
  const store = usePartyStore();

  if (isPreview) {
    return (
      <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
        <div className="mb-6">
          <span className="text-xs text-zinc-500">← 파티로</span>
          <h1 className="text-xl font-bold text-zinc-100 mt-1">미리보기님의 최종 추천</h1>
          <p className="text-sm text-zinc-400 mt-0.5">피드백이 반영된 맞춤 칵테일이에요</p>
          <span className="inline-block mt-2 px-2 py-0.5 rounded text-xs bg-amber-400/10 text-amber-400 border border-amber-400/20">
            디자인 미리보기 모드
          </span>
        </div>
        <div className="flex flex-col gap-5">
          <CocktailCard cocktail={MOCK_FINAL} stage="final" />
          <Card>
            <CardHeader><CardTitle>추천 변화</CardTitle></CardHeader>
            <CardBody>
              <div className="flex items-center gap-3">
                <div className="flex-1 bg-zinc-800 rounded-xl p-3 text-center">
                  <p className="text-2xl mb-1">🍊</p>
                  <p className="text-sm font-medium text-zinc-300">{MOCK_TASTING.name}</p>
                  <p className="text-xs text-zinc-500 mt-1">시음 추천</p>
                </div>
                <div className="flex flex-col items-center gap-1 text-zinc-500">
                  <span className="text-xl">→</span>
                  <span className="text-xs">피드백 반영</span>
                </div>
                <div className="flex-1 bg-amber-400/10 border border-amber-400/20 rounded-xl p-3 text-center">
                  <p className="text-2xl mb-1">🌸</p>
                  <p className="text-sm font-medium text-amber-300">{MOCK_FINAL.name}</p>
                  <p className="text-xs text-amber-500/70 mt-1">최종 추천</p>
                </div>
              </div>
            </CardBody>
          </Card>
          <Button size="lg" className="w-full" onClick={() => {}}>
            🍹 이 칵테일로 제조하기 →
          </Button>
        </div>
      </main>
    );
  }

  if (!guest?.finalRecommendation) {
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
      recipe: guest.finalRecommendation!.recipe ?? [],
      adjustments: guest.finalRecommendation!.adjustments ?? [],
      timestamp: now(),
      metadata: { temperature: '4°C', mixingSpeed: 'medium', glassType: 'highball' },
    };
    store.updateBrewState(id, guestId, { status: 'idle', progress: 0, devicePayload: payload });
    store.updateGuestStep(id, guestId, 'brew');
    router.push(`/party/${id}/guest/${guestId}/brew`);
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="mb-6">
        <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">← 파티로</Link>
        <h1 className="text-xl font-bold text-zinc-100 mt-1">{guest.name}님의 최종 추천</h1>
        <p className="text-sm text-zinc-400 mt-0.5">피드백이 반영된 맞춤 칵테일이에요</p>
      </div>

      <div className="flex flex-col gap-5">
        <CocktailCard cocktail={guest.finalRecommendation} stage="final" />

        {guest.tastingRecommendation && guest.tastingRecommendation.id !== guest.finalRecommendation.id && (
          <Card>
            <CardHeader><CardTitle>추천 변화</CardTitle></CardHeader>
            <CardBody>
              <div className="flex items-center gap-3">
                <div className="flex-1 bg-zinc-800 rounded-xl p-3 text-center">
                  <p className="text-2xl mb-1">🍹</p>
                  <p className="text-sm font-medium text-zinc-300">{guest.tastingRecommendation.name}</p>
                  <p className="text-xs text-zinc-500 mt-1">시음 추천</p>
                </div>
                <div className="flex flex-col items-center gap-1 text-zinc-500">
                  <span className="text-xl">→</span>
                  <span className="text-xs">피드백 반영</span>
                </div>
                <div className="flex-1 bg-amber-400/10 border border-amber-400/20 rounded-xl p-3 text-center">
                  <p className="text-2xl mb-1">🍹</p>
                  <p className="text-sm font-medium text-amber-300">{guest.finalRecommendation.name}</p>
                  <p className="text-xs text-amber-500/70 mt-1">최종 추천</p>
                </div>
              </div>
            </CardBody>
          </Card>
        )}

        <Button size="lg" className="w-full" onClick={handleBrew}>
          🍹 이 칵테일로 제조하기 →
        </Button>
      </div>
    </main>
  );
}
