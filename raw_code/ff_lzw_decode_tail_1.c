void ff_lzw_decode_tail(LZWState *p)

{

    struct LZWState *s = (struct LZWState *)p;

    while(!s->eob_reached)

        lzw_get_code(s);

}
