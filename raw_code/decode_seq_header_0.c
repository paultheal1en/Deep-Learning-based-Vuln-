static int decode_seq_header(AVSContext *h) {

    MpegEncContext *s = &h->s;

    int frame_rate_code;



    h->profile =         get_bits(&s->gb,8);

    h->level =           get_bits(&s->gb,8);

    skip_bits1(&s->gb); //progressive sequence

    s->width =           get_bits(&s->gb,14);

    s->height =          get_bits(&s->gb,14);

    skip_bits(&s->gb,2); //chroma format

    skip_bits(&s->gb,3); //sample_precision

    h->aspect_ratio =    get_bits(&s->gb,4);

    frame_rate_code =    get_bits(&s->gb,4);

    skip_bits(&s->gb,18);//bit_rate_lower

    skip_bits1(&s->gb);  //marker_bit

    skip_bits(&s->gb,12);//bit_rate_upper

    s->low_delay =       get_bits1(&s->gb);

    h->mb_width  = (s->width  + 15) >> 4;

    h->mb_height = (s->height + 15) >> 4;

    h->s.avctx->time_base.den = avpriv_frame_rate_tab[frame_rate_code].num;

    h->s.avctx->time_base.num = avpriv_frame_rate_tab[frame_rate_code].den;

    h->s.avctx->width  = s->width;

    h->s.avctx->height = s->height;

    if(!h->top_qp)

        ff_cavs_init_top_lines(h);

    return 0;

}
