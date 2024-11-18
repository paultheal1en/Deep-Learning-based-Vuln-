static int check_slice_end(RV34DecContext *r, MpegEncContext *s)

{

    int bits;

    if(s->mb_y >= s->mb_height)

        return 1;

    if(!s->mb_num_left)

        return 1;

    if(r->s.mb_skip_run > 1)

        return 0;

    bits = get_bits_left(&s->gb);

    if(bits < 0 || (bits < 8 && !show_bits(&s->gb, bits)))

        return 1;

    return 0;

}
