static void tcg_temp_free_internal(int idx)

{

    TCGContext *s = &tcg_ctx;

    TCGTemp *ts;

    int k;



#if defined(CONFIG_DEBUG_TCG)

    s->temps_in_use--;

    if (s->temps_in_use < 0) {

        fprintf(stderr, "More temporaries freed than allocated!\n");

    }

#endif



    assert(idx >= s->nb_globals && idx < s->nb_temps);

    ts = &s->temps[idx];

    assert(ts->temp_allocated != 0);

    ts->temp_allocated = 0;



    k = ts->base_type + (ts->temp_local ? TCG_TYPE_COUNT : 0);

    set_bit(idx, s->free_temps[k].l);

}
