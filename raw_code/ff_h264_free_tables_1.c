void ff_h264_free_tables(H264Context *h, int free_rbsp)

{

    int i;



    av_freep(&h->intra4x4_pred_mode);

    av_freep(&h->chroma_pred_mode_table);

    av_freep(&h->cbp_table);

    av_freep(&h->mvd_table[0]);

    av_freep(&h->mvd_table[1]);

    av_freep(&h->direct_table);

    av_freep(&h->non_zero_count);

    av_freep(&h->slice_table_base);

    h->slice_table = NULL;

    av_freep(&h->list_counts);



    av_freep(&h->mb2b_xy);

    av_freep(&h->mb2br_xy);



    av_buffer_pool_uninit(&h->qscale_table_pool);

    av_buffer_pool_uninit(&h->mb_type_pool);

    av_buffer_pool_uninit(&h->motion_val_pool);

    av_buffer_pool_uninit(&h->ref_index_pool);



    if (free_rbsp && h->DPB) {

        for (i = 0; i < H264_MAX_PICTURE_COUNT; i++)

            ff_h264_unref_picture(h, &h->DPB[i]);

        av_freep(&h->DPB);

    }



    h->cur_pic_ptr = NULL;



    for (i = 0; i < h->nb_slice_ctx; i++) {

        H264SliceContext *sl = &h->slice_ctx[i];



        av_freep(&sl->dc_val_base);

        av_freep(&sl->er.mb_index2xy);

        av_freep(&sl->er.error_status_table);

        av_freep(&sl->er.er_temp_buffer);



        av_freep(&sl->bipred_scratchpad);

        av_freep(&sl->edge_emu_buffer);

        av_freep(&sl->top_borders[0]);

        av_freep(&sl->top_borders[1]);



        sl->bipred_scratchpad_allocated = 0;

        sl->edge_emu_buffer_allocated   = 0;

        sl->top_borders_allocated[0]    = 0;

        sl->top_borders_allocated[1]    = 0;



        if (free_rbsp) {

            av_freep(&sl->rbsp_buffer);

            sl->rbsp_buffer_size            = 0;

        }

    }

}
