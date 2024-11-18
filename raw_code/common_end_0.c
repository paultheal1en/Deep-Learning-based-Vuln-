static void common_end(SnowContext *s){

    av_freep(&s->spatial_dwt_buffer);

    av_freep(&s->mb_band.buf);

    av_freep(&s->mv_band[0].buf);

    av_freep(&s->mv_band[1].buf);



    av_freep(&s->m.me.scratchpad);    

    av_freep(&s->m.me.map);

    av_freep(&s->m.me.score_map);

    av_freep(&s->mb_type);

    av_freep(&s->mb_mean);

    av_freep(&s->dummy);

    av_freep(&s->motion_val8);

    av_freep(&s->motion_val16);

}
