static void flush_dpb(AVCodecContext *avctx)

{

    H264Context *h = avctx->priv_data;

    int i;



    memset(h->delayed_pic, 0, sizeof(h->delayed_pic));



    ff_h264_flush_change(h);



    if (h->DPB)

        for (i = 0; i < H264_MAX_PICTURE_COUNT; i++)

            ff_h264_unref_picture(h, &h->DPB[i]);

    h->cur_pic_ptr = NULL;

    ff_h264_unref_picture(h, &h->cur_pic);



    h->mb_y = 0;



    ff_h264_free_tables(h);

    h->context_initialized = 0;

}
