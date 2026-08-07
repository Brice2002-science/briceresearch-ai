-- BRICERESEARCH AI — schéma Supabase (à coller dans SQL Editor puis "Run")
-- Deux tables + Row Level Security : chaque personne ne voit QUE ses propres données.

create table if not exists public.conversations (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  title      text,
  created_at timestamptz not null default now()
);

create table if not exists public.messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  user_id         uuid not null references auth.users(id) on delete cascade,
  role            text not null check (role in ('user','assistant')),
  content         text not null,
  created_at      timestamptz not null default now()
);

create index if not exists idx_conv_user on public.conversations(user_id);
create index if not exists idx_msg_conv on public.messages(conversation_id);

alter table public.conversations enable row level security;
alter table public.messages     enable row level security;

-- Politiques : accès limité au propriétaire (auth.uid())
create policy "own conversations - select" on public.conversations
  for select using (auth.uid() = user_id);
create policy "own conversations - insert" on public.conversations
  for insert with check (auth.uid() = user_id);
create policy "own conversations - delete" on public.conversations
  for delete using (auth.uid() = user_id);

create policy "own messages - select" on public.messages
  for select using (auth.uid() = user_id);
create policy "own messages - insert" on public.messages
  for insert with check (auth.uid() = user_id);
create policy "own messages - delete" on public.messages
  for delete using (auth.uid() = user_id);
