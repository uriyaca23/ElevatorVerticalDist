import Link from "next/link";

export function AppHeader({ children }: { children?: React.ReactNode }) {
  return (
    <header className="flex items-center gap-4 border-b border-neutral-200 bg-white px-4 py-3 sm:px-6">
      <Link href="/experiments" className="font-semibold">
        Shaft Lab
      </Link>
      <div className="min-w-0 flex-1 truncate text-sm text-neutral-500">{children}</div>
      <form action="/api/logout" method="post">
        <button className="text-sm text-neutral-500 hover:text-neutral-900">Log out</button>
      </form>
    </header>
  );
}
