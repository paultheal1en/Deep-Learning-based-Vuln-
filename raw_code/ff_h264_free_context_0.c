av_cold void ff_h264_free_context(H264Context *h)

{

    int i;



    ff_h264_free_tables(h);



    if (h->DPB) {

        for (i = 0; i < H264_MAX_PICTURE_COUNT; i++)

            ff_h264_unref_picture(h, &h->DPB[i]);

        av_freep(&h->DPB);

    }



    h->cur_pic_ptr = NULL;



    for (i = 0; i < h->nb_slice_ctx; i++)

        av_freep(&h->slice_ctx[i].rbsp_buffer);

    av_freep(&h->slice_ctx);

    h->nb_slice_ctx = 0;



    for (i = 0; i < MAX_SPS_COUNT; i++)

        av_freep(h->sps_buffers + i);



    for (i = 0; i < MAX_PPS_COUNT; i++)

        av_freep(h->pps_buffers + i);

}
