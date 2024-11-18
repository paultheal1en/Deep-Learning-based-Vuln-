static void mpeg4_encode_gop_header(MpegEncContext *s)

{

    int hours, minutes, seconds;

    int64_t time;



    put_bits(&s->pb, 16, 0);

    put_bits(&s->pb, 16, GOP_STARTCODE);



    time = s->current_picture_ptr->f.pts;

    if (s->reordered_input_picture[1])

        time = FFMIN(time, s->reordered_input_picture[1]->f.pts);

    time = time * s->avctx->time_base.num;



    seconds  = time / s->avctx->time_base.den;

    minutes  = seconds / 60;

    seconds %= 60;

    hours    = minutes / 60;

    minutes %= 60;

    hours   %= 24;



    put_bits(&s->pb, 5, hours);

    put_bits(&s->pb, 6, minutes);

    put_bits(&s->pb, 1, 1);

    put_bits(&s->pb, 6, seconds);



    put_bits(&s->pb, 1, !!(s->flags & CODEC_FLAG_CLOSED_GOP));

    put_bits(&s->pb, 1, 0);  // broken link == NO



    s->last_time_base = time / s->avctx->time_base.den;



    ff_mpeg4_stuffing(&s->pb);

}
