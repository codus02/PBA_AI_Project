import type { CocktailRecommendation } from '@/lib/types';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';

interface CocktailCardProps {
  cocktail: CocktailRecommendation;
  stage?: 'tasting' | 'final';
}

export default function CocktailCard({ cocktail, stage = 'tasting' }: CocktailCardProps) {
  return (
    <Card glow className="flex flex-col gap-4">
      <div className="flex items-start gap-4">
        <div className="text-6xl">{cocktail.imageEmoji}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {stage === 'final' && (
              <Badge variant="amber">최종 추천</Badge>
            )}
            {stage === 'tasting' && (
              <Badge variant="blue">시음 추천</Badge>
            )}
          </div>
          <h2 className="text-xl font-bold text-zinc-100">{cocktail.name}</h2>
          <p className="text-sm text-zinc-400 mt-1 leading-relaxed">{cocktail.description}</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {cocktail.tags.map((tag) => (
          <Badge key={tag} variant="default">#{tag}</Badge>
        ))}
      </div>

      <div className="bg-zinc-800/60 rounded-xl p-4">
        <p className="text-xs text-zinc-500 mb-1">추천 이유</p>
        <p className="text-sm text-zinc-300 leading-relaxed">{cocktail.reason}</p>
      </div>

      <div>
        <p className="text-xs text-zinc-500 mb-2">레시피</p>
        <div className="flex flex-col gap-1.5">
          {cocktail.recipe.map((item, i) => (
            <div key={i} className="flex justify-between items-center py-1.5 border-b border-zinc-800 last:border-0">
              <span className="text-sm text-zinc-300">{item.ingredient}</span>
              <span className="text-sm text-amber-400 font-medium">
                {item.amount} {item.unit}
              </span>
            </div>
          ))}
        </div>
      </div>

      {cocktail.adjustments && cocktail.adjustments.length > 0 && (
        <div className="bg-emerald-400/10 border border-emerald-400/20 rounded-xl p-4">
          <p className="text-xs text-emerald-400 mb-2">피드백 반영 조정사항</p>
          <div className="flex flex-col gap-1.5">
            {cocktail.adjustments.map((adj, i) => (
              <div key={i} className="flex gap-2 text-sm">
                <span className="text-emerald-400">↗</span>
                <span className="text-zinc-300">
                  <span className="font-medium">{adj.ingredient}</span>
                  {' '}<span className="text-zinc-400">{adj.change}</span>
                  {' — '}<span className="text-zinc-500 text-xs">{adj.reason}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
