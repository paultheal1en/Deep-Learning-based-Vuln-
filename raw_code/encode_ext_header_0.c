static int encode_ext_header(Wmv2Context *w){

    MpegEncContext * const s= &w->s;

    PutBitContext pb;

    int code;



    init_put_bits(&pb, s->avctx->extradata, s->avctx->extradata_size);



    put_bits(&pb, 5, s->avctx->time_base.den / s->avctx->time_base.num); //yes 29.97 -> 29

    put_bits(&pb, 11, FFMIN(s->bit_rate/1024, 2047));



    put_bits(&pb, 1, w->mspel_bit=1);

    put_bits(&pb, 1, w->flag3=1);

    put_bits(&pb, 1, w->abt_flag=1);

    put_bits(&pb, 1, w->j_type_bit=1);

    put_bits(&pb, 1, w->top_left_mv_flag=0);

    put_bits(&pb, 1, w->per_mb_rl_bit=1);

    put_bits(&pb, 3, code=1);



    flush_put_bits(&pb);



    s->slice_height = s->mb_height / code;



    return 0;

}
