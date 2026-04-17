import type { CocktailRecommendation } from '@/lib/types';

export const MOCK_COCKTAILS: CocktailRecommendation[] = [
  {
    id: 'moscow-mule',
    name: 'Moscow Mule',
    description: '진저비어의 청량감과 라임의 상큼함이 어우러진 클래식 칵테일',
    reason: '가벼운 산미와 상쾌함을 선호하는 취향에 잘 맞습니다. 알코올 강도도 적당해 부담 없이 즐길 수 있어요.',
    recipe: [
      { ingredient: '보드카', amount: 45, unit: 'ml' },
      { ingredient: '진저비어', amount: 120, unit: 'ml' },
      { ingredient: '라임 주스', amount: 15, unit: 'ml' },
      { ingredient: '얼음', amount: 5, unit: '개' },
    ],
    imageEmoji: '🍹',
    tags: ['상큼', '청량', '클래식', '가벼움'],
  },
  {
    id: 'espresso-martini',
    name: 'Espresso Martini',
    description: '에스프레소의 깊은 향과 부드러운 커피 리큐어가 조화를 이루는 세련된 칵테일',
    reason: '쓴맛과 진한 향을 즐기는 취향에 적합합니다. 파티 분위기를 고급스럽게 끌어올려 줄 선택이에요.',
    recipe: [
      { ingredient: '보드카', amount: 45, unit: 'ml' },
      { ingredient: '에스프레소', amount: 30, unit: 'ml' },
      { ingredient: '커피 리큐어', amount: 15, unit: 'ml' },
      { ingredient: '심플 시럽', amount: 10, unit: 'ml' },
    ],
    imageEmoji: '☕',
    tags: ['커피', '쌉쌀', '세련된', '강함'],
  },
  {
    id: 'aperol-spritz',
    name: 'Aperol Spritz',
    description: '이탈리아 아페롤의 달콤 쌉쌀한 맛과 프로세코의 기포가 만나는 여름 칵테일',
    reason: '달콤하면서도 쓴맛이 조화로운 음료를 선호하는 분께 추천합니다. 가벼운 알코올 도수로 오래 즐길 수 있어요.',
    recipe: [
      { ingredient: '아페롤', amount: 60, unit: 'ml' },
      { ingredient: '프로세코', amount: 90, unit: 'ml' },
      { ingredient: '탄산수', amount: 30, unit: 'ml' },
      { ingredient: '오렌지 슬라이스', amount: 1, unit: '개' },
    ],
    imageEmoji: '🍊',
    tags: ['달콤쌉쌀', '기포', '과일향', '가벼움'],
  },
  {
    id: 'old-fashioned',
    name: 'Old Fashioned',
    description: '버번 위스키의 깊은 풍미와 비터스의 복잡한 아로마가 어우러진 클래식 중의 클래식',
    reason: '강하고 깊은 풍미를 원하는 경험자에게 어울립니다. 우디하고 복잡한 향을 즐기는 분께 최적이에요.',
    recipe: [
      { ingredient: '버번 위스키', amount: 60, unit: 'ml' },
      { ingredient: '앙고스투라 비터스', amount: 2, unit: 'dash' },
      { ingredient: '설탕 시럽', amount: 5, unit: 'ml' },
      { ingredient: '오렌지 필', amount: 1, unit: '조각' },
    ],
    imageEmoji: '🥃',
    tags: ['위스키', '강함', '클래식', '우디'],
  },
  {
    id: 'hugo-spritz',
    name: 'Hugo Spritz',
    description: '엘더플라워 코디얼의 꽃향기와 민트, 프로세코가 어우러진 상쾌하고 향긋한 칵테일',
    reason: '꽃향기와 허브 향을 좋아하는 분께 잘 맞습니다. 화사한 파티 분위기에 어울리는 가벼운 선택이에요.',
    recipe: [
      { ingredient: '프로세코', amount: 90, unit: 'ml' },
      { ingredient: '엘더플라워 코디얼', amount: 20, unit: 'ml' },
      { ingredient: '탄산수', amount: 30, unit: 'ml' },
      { ingredient: '민트', amount: 3, unit: '잎' },
      { ingredient: '라임', amount: 2, unit: '슬라이스' },
    ],
    imageEmoji: '🌸',
    tags: ['꽃향기', '허브', '청량', '가벼움'],
  },
  {
    id: 'margarita',
    name: 'Margarita',
    description: '테킬라의 강렬함과 트리플섹의 오렌지 향, 라임의 산미가 균형 잡힌 멕시칸 클래식',
    reason: '새콤하고 강한 맛을 즐기는 분께 딱입니다. 소금 림의 짭조름한 대비가 매력적이에요.',
    recipe: [
      { ingredient: '테킬라', amount: 45, unit: 'ml' },
      { ingredient: '트리플섹', amount: 30, unit: 'ml' },
      { ingredient: '라임 주스', amount: 20, unit: 'ml' },
      { ingredient: '소금', amount: 1, unit: '핀치' },
    ],
    imageEmoji: '🍋',
    tags: ['새콤', '강함', '클래식', '멕시칸'],
  },
  {
    id: 'virgin-mojito',
    name: 'Virgin Mojito',
    description: '민트와 라임, 탄산수로 만드는 논알코올 모히토. 상쾌함은 그대로 알코올은 없이',
    reason: '알코올을 원하지 않거나 첫 경험인 분께 완벽한 선택입니다. 파티를 충분히 즐길 수 있어요.',
    recipe: [
      { ingredient: '민트 잎', amount: 8, unit: '잎' },
      { ingredient: '라임 주스', amount: 30, unit: 'ml' },
      { ingredient: '설탕 시럽', amount: 15, unit: 'ml' },
      { ingredient: '탄산수', amount: 150, unit: 'ml' },
      { ingredient: '얼음', amount: 5, unit: '개' },
    ],
    imageEmoji: '🌿',
    tags: ['논알코올', '민트', '상쾌', '청량'],
  },
  {
    id: 'whisky-sour',
    name: 'Whisky Sour',
    description: '위스키의 풍미에 레몬의 산미와 달콤함이 더해진 균형 잡힌 칵테일',
    reason: '중간 강도의 위스키 풍미를 즐기면서 산미가 있는 음료를 원하는 분께 추천합니다.',
    recipe: [
      { ingredient: '버번 위스키', amount: 45, unit: 'ml' },
      { ingredient: '레몬 주스', amount: 30, unit: 'ml' },
      { ingredient: '심플 시럽', amount: 15, unit: 'ml' },
      { ingredient: '달걀흰자', amount: 15, unit: 'ml' },
    ],
    imageEmoji: '🍋',
    tags: ['새콤', '위스키', '균형', '중간'],
  },
];

export function getRandomCocktail(excludeId?: string): CocktailRecommendation {
  const available = excludeId
    ? MOCK_COCKTAILS.filter((c) => c.id !== excludeId)
    : MOCK_COCKTAILS;
  return available[Math.floor(Math.random() * available.length)];
}

export function getCocktailById(id: string): CocktailRecommendation | undefined {
  return MOCK_COCKTAILS.find((c) => c.id === id);
}
