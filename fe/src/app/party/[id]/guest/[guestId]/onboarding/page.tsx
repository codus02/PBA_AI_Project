'use client';

import { use, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { getInitialRecommendation } from '@/lib/mock/recommendations';
import { getFollowUpQuestions } from '@/lib/mock/questions';
import type { GuestPreferences, FollowUpAnswer } from '@/lib/types';
import { now } from '@/lib/utils';
import TasteForm from '@/components/onboarding/TasteForm';
import FollowUpQuestions from '@/components/onboarding/FollowUpQuestions';
import StepIndicator from '@/components/ui/StepIndicator';

const STEPS = [
  { label: '취향', icon: '🎯' },
  { label: '추가질문', icon: '💬' },
];

export default function OnboardingPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const guest = useGuest(id, guestId);
  const store = usePartyStore();
  const [step, setStep] = useState(0);
  const [followUpQuestions] = useState(() => getFollowUpQuestions(3));

  if (!guest) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400">게스트를 찾을 수 없어요</p>
          <Link href={`/party/${id}`} className="text-amber-400 text-sm mt-2 inline-block">
            파티로 돌아가기
          </Link>
        </div>
      </main>
    );
  }

  const handleTasteSubmit = (prefs: GuestPreferences) => {
    store.setPreferences(id, guestId, prefs);
    store.addConversationEntry(id, guestId, {
      timestamp: now(),
      type: 'preference_input',
      content: `취향 입력 완료: 단맛 ${prefs.sweetness}, 신맛 ${prefs.sourness}, 강도 ${prefs.intensity}`,
    });
    setStep(1);
  };

  const handleFollowUpSubmit = (answers: FollowUpAnswer[]) => {
    store.setFollowUpAnswers(id, guestId, answers);
    answers.forEach((a) => {
      store.addConversationEntry(id, guestId, {
        timestamp: now(),
        type: 'followup_a',
        content: `${a.question} → ${a.answer}`,
      });
    });

    const prefs = store.getGuest(id, guestId)?.preferences;
    if (prefs) {
      const recommendation = getInitialRecommendation(prefs);
      store.setTastingRecommendation(id, guestId, recommendation);
      store.addRecommendationEntry(id, guestId, {
        timestamp: now(),
        stage: 'tasting',
        cocktailId: recommendation.id,
        cocktailName: recommendation.name,
      });
    }
    router.push(`/party/${id}/guest/${guestId}/tasting`);
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
            ← 파티로
          </Link>
          <h1 className="text-xl font-bold text-zinc-100 mt-1">
            {guest.name}님의 취향 입력
          </h1>
        </div>
      </div>

      <div className="flex justify-center mb-8">
        <StepIndicator steps={STEPS} current={step} />
      </div>

      {step === 0 && <TasteForm onSubmit={handleTasteSubmit} />}
      {step === 1 && (
        <FollowUpQuestions
          questions={followUpQuestions}
          onSubmit={handleFollowUpSubmit}
        />
      )}
    </main>
  );
}
