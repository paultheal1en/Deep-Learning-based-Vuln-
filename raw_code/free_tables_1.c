static void free_tables(H264Context *h){

    int i;

    H264Context *hx;

    av_freep(&h->intra4x4_pred_mode);

    av_freep(&h->chroma_pred_mode_table);

    av_freep(&h->cbp_table);

    av_freep(&h->mvd_table[0]);

    av_freep(&h->mvd_table[1]);

    av_freep(&h->direct_table);

    av_freep(&h->non_zero_count);

    av_freep(&h->slice_table_base);

    h->slice_table= NULL;

    av_freep(&h->list_counts);



    av_freep(&h->mb2b_xy);

    av_freep(&h->mb2br_xy);



    for(i = 0; i < MAX_THREADS; i++) {

        hx = h->thread_context[i];

        if(!hx) continue;

        av_freep(&hx->top_borders[1]);

        av_freep(&hx->top_borders[0]);

        av_freep(&hx->s.obmc_scratchpad);

        av_freep(&hx->rbsp_buffer[1]);

        av_freep(&hx->rbsp_buffer[0]);

        hx->rbsp_buffer_size[0] = 0;

        hx->rbsp_buffer_size[1] = 0;

        if (i) av_freep(&h->thread_context[i]);

    }

}
