static char *tcg_get_arg_str_idx(TCGContext *s, char *buf,

                                 int buf_size, int idx)

{

    assert(idx >= 0 && idx < s->nb_temps);

    return tcg_get_arg_str_ptr(s, buf, buf_size, &s->temps[idx]);

}
