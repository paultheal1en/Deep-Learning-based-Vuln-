void ff_h261_loop_filter(MpegEncContext *s){

    H261Context * h= (H261Context*)s;

    const int linesize  = s->linesize;

    const int uvlinesize= s->uvlinesize;

    uint8_t *dest_y = s->dest[0];

    uint8_t *dest_cb= s->dest[1];

    uint8_t *dest_cr= s->dest[2];



    if(!(IS_FIL (h->mtype)))

        return;



    s->dsp.h261_loop_filter(dest_y                   , linesize);

    s->dsp.h261_loop_filter(dest_y                + 8, linesize);

    s->dsp.h261_loop_filter(dest_y + 8 * linesize    , linesize);

    s->dsp.h261_loop_filter(dest_y + 8 * linesize + 8, linesize);

    s->dsp.h261_loop_filter(dest_cb, uvlinesize);

    s->dsp.h261_loop_filter(dest_cr, uvlinesize);

}
