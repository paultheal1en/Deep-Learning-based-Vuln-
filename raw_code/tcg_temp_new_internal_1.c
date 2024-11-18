static inline int tcg_temp_new_internal(TCGType type, int temp_local)
{
    TCGContext *s = &tcg_ctx;
    TCGTemp *ts;
    int idx, k;
    k = type;
    if (temp_local)
        k += TCG_TYPE_COUNT;
    idx = s->first_free_temp[k];
    if (idx != -1) {
        /* There is already an available temp with the
           right type */
        ts = &s->temps[idx];
        s->first_free_temp[k] = ts->next_free_temp;
        ts->temp_allocated = 1;
        assert(ts->temp_local == temp_local);
    } else {
        idx = s->nb_temps;
#if TCG_TARGET_REG_BITS == 32
        if (type == TCG_TYPE_I64) {
            tcg_temp_alloc(s, s->nb_temps + 2);
            ts = &s->temps[s->nb_temps];
            ts->base_type = type;
            ts->type = TCG_TYPE_I32;
            ts->temp_allocated = 1;
            ts->temp_local = temp_local;
            ts->name = NULL;
            ts++;
            ts->base_type = TCG_TYPE_I32;
            ts->type = TCG_TYPE_I32;
            ts->temp_allocated = 1;
            ts->temp_local = temp_local;
            ts->name = NULL;
            s->nb_temps += 2;
        } else
        {
            tcg_temp_alloc(s, s->nb_temps + 1);
            ts = &s->temps[s->nb_temps];
            ts->base_type = type;
            ts->type = type;
            ts->temp_allocated = 1;
            ts->temp_local = temp_local;
            ts->name = NULL;
            s->nb_temps++;
        }
    }
    return idx;
}