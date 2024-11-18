static void switch_buffer(MPADecodeContext *s, int *pos, int *end_pos,

                          int *end_pos2)

{

    if (s->in_gb.buffer && *pos >= s->gb.size_in_bits) {

        s->gb           = s->in_gb;

        s->in_gb.buffer = NULL;

        assert((get_bits_count(&s->gb) & 7) == 0);

        skip_bits_long(&s->gb, *pos - *end_pos);

        *end_pos2 =

        *end_pos  = *end_pos2 + get_bits_count(&s->gb) - *pos;

        *pos      = get_bits_count(&s->gb);

    }

}
