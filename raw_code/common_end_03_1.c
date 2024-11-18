static av_cold int common_end(AVCodecContext *avctx){
    FFV1Context *s = avctx->priv_data;
    int i, j;
    for(j=0; j<s->slice_count; j++){
        FFV1Context *fs= s->slice_context[j];
        for(i=0; i<s->plane_count; i++){
            PlaneContext *p= &fs->plane[i];
            av_freep(&p->state);
            av_freep(&p->vlc_state);
        av_freep(&fs->sample_buffer);
    av_freep(&avctx->stats_out);
    for(j=0; j<s->quant_table_count; j++){
        av_freep(&s->initial_states[j]);
            FFV1Context *sf= s->slice_context[i];
            av_freep(&sf->rc_stat2[j]);
        av_freep(&s->rc_stat2[j]);
    return 0;