static bool use_exit_tb(DisasContext *s)

{

    return (s->singlestep_enabled ||

            (s->tb->cflags & CF_LAST_IO) ||

            (s->tb->flags & FLAG_MASK_PER));

}
