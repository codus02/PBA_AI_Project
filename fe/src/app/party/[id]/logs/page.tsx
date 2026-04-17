'use client';

import { use } from 'react';
import Link from 'next/link';
import { useParty } from '@/lib/store/partyStore';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { formatTime } from '@/lib/utils';

export default function LogsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const party = useParty(id);

  if (!party) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <p className="text-zinc-400">파티를 찾을 수 없어요</p>
      </main>
    );
  }

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="mb-6">
        <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
          ← 파티로
        </Link>
        <h1 className="text-xl font-bold text-zinc-100 mt-1">세션 로그</h1>
        <p className="text-sm text-zinc-400 mt-0.5">{party.name}</p>
      </div>

      {party.guests.length === 0 ? (
        <Card>
          <CardBody className="text-center py-8">
            <p className="text-zinc-500">아직 로그가 없어요</p>
          </CardBody>
        </Card>
      ) : (
        <div className="flex flex-col gap-6">
          {party.guests.map((guest) => (
            <div key={guest.id} className="flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <h2 className="text-base font-semibold text-zinc-200">{guest.name}</h2>
                {guest.satisfaction && (
                  <span className="text-xs text-amber-400">
                    {'⭐'.repeat(guest.satisfaction)} ({guest.satisfaction}/5)
                  </span>
                )}
              </div>

              {/* 추천 로그 */}
              {guest.logs.recommendationLog.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">🍹 추천 로그</CardTitle>
                  </CardHeader>
                  <CardBody className="flex flex-col gap-2">
                    {guest.logs.recommendationLog.map((entry) => (
                      <div key={entry.id} className="flex items-center gap-3 text-sm">
                        <span className="text-zinc-600 text-xs min-w-[50px]">
                          {formatTime(entry.timestamp)}
                        </span>
                        <Badge variant={entry.stage === 'tasting' ? 'blue' : 'amber'}>
                          {entry.stage === 'tasting' ? '시음' : '최종'}
                        </Badge>
                        <span className="text-zinc-300">{entry.cocktailName}</span>
                      </div>
                    ))}
                  </CardBody>
                </Card>
              )}

              {/* 피드백 로그 */}
              {guest.logs.feedbackLog.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">💬 피드백 로그</CardTitle>
                  </CardHeader>
                  <CardBody className="flex flex-col gap-3">
                    {guest.logs.feedbackLog.map((entry) => (
                      <div key={entry.id} className="flex flex-col gap-2">
                        <div className="flex gap-2 text-sm">
                          <span className="text-zinc-600 text-xs min-w-[50px]">
                            {formatTime(entry.timestamp)}
                          </span>
                          <span className="text-zinc-300 italic">"{entry.rawFeedback}"</span>
                        </div>
                        <div className="flex flex-wrap gap-1.5 pl-14">
                          {entry.adjustments.map((adj, i) => (
                            <Badge
                              key={i}
                              variant={adj.direction === 'increase' ? 'green' : 'red'}
                            >
                              {adj.direction === 'increase' ? '↑' : '↓'} {adj.label}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </CardBody>
                </Card>
              )}

              {/* 대화 로그 */}
              {guest.logs.conversationLog.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">📝 대화 로그</CardTitle>
                  </CardHeader>
                  <CardBody className="flex flex-col gap-2">
                    {guest.logs.conversationLog.map((entry) => (
                      <div key={entry.id} className="flex gap-3 text-xs">
                        <span className="text-zinc-600 min-w-[50px]">
                          {formatTime(entry.timestamp)}
                        </span>
                        <span className="text-zinc-400">{entry.content}</span>
                      </div>
                    ))}
                  </CardBody>
                </Card>
              )}

              {guest.finalRecommendation && (
                <div className="bg-amber-400/10 border border-amber-400/20 rounded-xl p-4 flex items-center gap-3">
                  <span className="text-3xl">{guest.finalRecommendation.imageEmoji}</span>
                  <div>
                    <p className="text-sm font-medium text-amber-300">최종 칵테일</p>
                    <p className="text-zinc-300">{guest.finalRecommendation.name}</p>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
