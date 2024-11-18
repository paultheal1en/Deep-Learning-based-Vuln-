void ff_h261_loop_filter(H261Context * h){

    MpegEncContext * const s = &h->s;

    int i;

    const int linesize  = s->linesize;

    const int uvlinesize= s->uvlinesize;

    uint8_t *dest_y = s->dest[0];

    uint8_t *dest_cb= s->dest[1];

    uint8_t *dest_cr= s->dest[2];

    uint8_t *src;



    CHECKED_ALLOCZ((src),sizeof(uint8_t) * 64 );



    for(i=0; i<8;i++)

        memcpy(src+i*8,dest_y+i*linesize,sizeof(uint8_t) * 8 );

    s->dsp.h261_v_loop_filter(dest_y, src, linesize);

    s->dsp.h261_h_loop_filter(dest_y, src, linesize);



    for(i=0; i<8;i++)

        memcpy(src+i*8,dest_y+i*linesize + 8,sizeof(uint8_t) * 8 );

    s->dsp.h261_v_loop_filter(dest_y + 8, src, linesize);

    s->dsp.h261_h_loop_filter(dest_y + 8, src, linesize);



    for(i=0; i<8;i++)

        memcpy(src+i*8,dest_y+(i+8)*linesize,sizeof(uint8_t) * 8 );

    s->dsp.h261_v_loop_filter(dest_y + 8 * linesize, src, linesize);

    s->dsp.h261_h_loop_filter(dest_y + 8 * linesize, src, linesize);



    for(i=0; i<8;i++)

        memcpy(src+i*8,dest_y+(i+8)*linesize + 8,sizeof(uint8_t) * 8 );

    s->dsp.h261_v_loop_filter(dest_y + 8 * linesize + 8, src, linesize);

    s->dsp.h261_h_loop_filter(dest_y + 8 * linesize + 8, src, linesize);



    for(i=0; i<8;i++)

        memcpy(src+i*8,dest_cb+i*uvlinesize,sizeof(uint8_t) * 8 );

    s->dsp.h261_v_loop_filter(dest_cb, src, uvlinesize);

    s->dsp.h261_h_loop_filter(dest_cb, src, uvlinesize);



    for(i=0; i<8;i++)

        memcpy(src+i*8,dest_cr+i*uvlinesize,sizeof(uint8_t) * 8 );

    s->dsp.h261_v_loop_filter(dest_cr, src, uvlinesize);

    s->dsp.h261_h_loop_filter(dest_cr, src, uvlinesize);



fail:

    av_free(src);



    return;

}
