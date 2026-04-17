import Link from 'next/link';
import { cn } from '@/lib/utils';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  backHref?: string;
  backLabel?: string;
  right?: React.ReactNode;
}

export default function PageHeader({ title, subtitle, backHref, backLabel = '뒤로', right }: PageHeaderProps) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6">
      <div className="flex items-start gap-3">
        {backHref && (
          <Link
            href={backHref}
            className="mt-0.5 text-zinc-400 hover:text-zinc-200 transition-colors text-sm"
          >
            ← {backLabel}
          </Link>
        )}
        {!backHref && (
          <div>
            <h1 className="text-2xl font-bold text-zinc-100">{title}</h1>
            {subtitle && <p className="text-sm text-zinc-400 mt-0.5">{subtitle}</p>}
          </div>
        )}
        {backHref && (
          <div>
            <h1 className="text-2xl font-bold text-zinc-100">{title}</h1>
            {subtitle && <p className="text-sm text-zinc-400 mt-0.5">{subtitle}</p>}
          </div>
        )}
      </div>
      {right && <div>{right}</div>}
    </div>
  );
}
