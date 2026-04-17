import type { FeedbackAnalysis, CocktailRecommendation } from '@/lib/types';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';

interface ExplanationPanelProps {
  initial: CocktailRecommendation;
  final: CocktailRecommendation;
  feedbackAnalysis: FeedbackAnalysis;
}

const DIMENSION_LABELS = {
  sweetness: '단맛',
  sourness: '신맛',
  bitterness: '쓴맛',
  strength: '도수',
  aroma: '향',
};

export default function ExplanationPanel({ initial, final, feedbackAnalysis }: ExplanationPanelProps) {
  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>피드백 분석 결과</CardTitle>
        </CardHeader>
        <CardBody className="flex flex-col gap-3">
          <div className="bg-zinc-800/60 rounded-xl p-3">
            <p className="text-xs text-zinc-500 mb-1">원본 피드백</p>
            <p className="text-sm text-zinc-300 italic">"{feedbackAnalysis.originalText}"</p>
          </div>

          <div className="flex flex-col gap-2">
            {feedbackAnalysis.adjustments.map((adj, i) => (
              <div key={i} className="flex items-center gap-3 py-2 border-b border-zinc-800 last:border-0">
                <div className={`text-lg ${adj.direction === 'increase' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {adj.direction === 'increase' ? '↑' : '↓'}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-zinc-200">
                    {DIMENSION_LABELS[adj.dimension] ?? adj.dimension}
                  </p>
                  <p className="text-xs text-zinc-500">{adj.label}</p>
                </div>
                <div className="text-right">
                  <div className="w-16 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${adj.direction === 'increase' ? 'bg-emerald-400' : 'bg-red-400'}`}
                      style={{ width: `${adj.magnitude * 100}%` }}
                    />
                  </div>
                  <p className="text-xs text-zinc-500 mt-0.5">{Math.round(adj.magnitude * 100)}%</p>
                </div>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>추천 변화 과정</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="flex items-center gap-3">
            <div className="flex-1 bg-zinc-800 rounded-xl p-3 text-center">
              <p className="text-2xl mb-1">{initial.imageEmoji}</p>
              <p className="text-sm font-medium text-zinc-300">{initial.name}</p>
              <Badge variant="blue" className="mt-1">시음 추천</Badge>
            </div>
            <div className="flex flex-col items-center gap-1 text-zinc-500">
              <span className="text-xl">→</span>
              <span className="text-xs">피드백 반영</span>
            </div>
            <div className="flex-1 bg-amber-400/10 border border-amber-400/20 rounded-xl p-3 text-center">
              <p className="text-2xl mb-1">{final.imageEmoji}</p>
              <p className="text-sm font-medium text-amber-300">{final.name}</p>
              <Badge variant="amber" className="mt-1">최종 추천</Badge>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
