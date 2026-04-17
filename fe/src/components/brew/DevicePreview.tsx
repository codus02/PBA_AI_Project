'use client';

import { useState } from 'react';
import type { DevicePayload } from '@/lib/types';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Button from '@/components/ui/Button';

interface DevicePreviewProps {
  payload: DevicePayload;
}

export default function DevicePreview({ payload }: DevicePreviewProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const jsonStr = JSON.stringify(payload, null, 2);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // return (
//     <Card>
//       <CardHeader>
//         <div className="flex items-center justify-between">
//           <CardTitle>장치 제어 페이로드</CardTitle>
//           <button
//             type="button"
//             onClick={() => setExpanded((v) => !v)}
//             className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
//           >
//             {expanded ? '접기 ▲' : '펼치기 ▼'}
//           </button>
//         </div>
//         <p className="text-xs text-zinc-500 mt-1">실제 하드웨어 연결 시 이 JSON이 장치로 전송됩니다</p>
//       </CardHeader>
//       <CardBody>
//         <div className="flex gap-2 mb-3">
//           <div className="flex-1 bg-zinc-800 rounded-lg px-3 py-2">
//             <p className="text-xs text-zinc-500">칵테일</p>
//             <p className="text-sm text-zinc-100 font-medium">{payload.cocktailName}</p>
//           </div>
//           <div className="flex-1 bg-zinc-800 rounded-lg px-3 py-2">
//             <p className="text-xs text-zinc-500">타임스탬프</p>
//             <p className="text-sm text-zinc-100 font-medium">
//               {new Date(payload.timestamp).toLocaleTimeString('ko-KR')}
//             </p>
//           </div>
//         </div>

//         {expanded && (
//           <div className="relative">
//             <pre className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 text-xs text-zinc-300 overflow-auto max-h-64 leading-relaxed">
//               {jsonStr}
//             </pre>
//             <Button
//               size="sm"
//               variant="secondary"
//               className="absolute top-2 right-2"
//               onClick={handleCopy}
//             >
//               {copied ? '✓ 복사됨' : '복사'}
//             </Button>
//           </div>
//         )}

//         {!expanded && (
//           <div className="flex flex-col gap-1">
//             {payload.recipe.slice(0, 3).map((item, i) => (
//               <div key={i} className="flex justify-between text-sm">
//                 <span className="text-zinc-400">{item.ingredient}</span>
//                 <span className="text-amber-400">{item.amount}{item.unit}</span>
//               </div>
//             ))}
//             {payload.recipe.length > 3 && (
//               <p className="text-xs text-zinc-600">+{payload.recipe.length - 3}개 더...</p>
//             )}
//           </div>
//         )}
//       </CardBody>
//     </Card>
//   );
}
