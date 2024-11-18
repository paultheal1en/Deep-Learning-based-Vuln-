static inline void tcg_temp_free_internal(int idx)
{
    TCGContext *s = &tcg_ctx;
    TCGTemp *ts;
    int k;
    assert(idx >= s->nb_globals && idx < s->nb_temps);
    ts = &s->temps[idx];
    assert(ts->temp_allocated != 0);
    ts->temp_allocated = 0;
    k = ts->base_type;
    if (ts->temp_local)
        k += TCG_TYPE_COUNT;
    ts->next_free_temp = s->first_free_temp[k];
    s->first_free_temp[k] = idx;