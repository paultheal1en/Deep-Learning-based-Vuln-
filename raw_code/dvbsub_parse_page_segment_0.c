static void dvbsub_parse_page_segment(AVCodecContext *avctx,

                                        const uint8_t *buf, int buf_size)

{

    DVBSubContext *ctx = avctx->priv_data;

    DVBSubRegionDisplay *display;

    DVBSubRegionDisplay *tmp_display_list, **tmp_ptr;



    const uint8_t *buf_end = buf + buf_size;

    int region_id;

    int page_state;



    if (buf_size < 1)

        return;



    ctx->time_out = *buf++;

    page_state = ((*buf++) >> 2) & 3;



    av_dlog(avctx, "Page time out %ds, state %d\n", ctx->time_out, page_state);



    if (page_state == 2) {

        delete_state(ctx);

    }



    tmp_display_list = ctx->display_list;

    ctx->display_list = NULL;

    ctx->display_list_size = 0;



    while (buf + 5 < buf_end) {

        region_id = *buf++;

        buf += 1;



        display = tmp_display_list;

        tmp_ptr = &tmp_display_list;



        while (display && display->region_id != region_id) {

            tmp_ptr = &display->next;

            display = display->next;

        }



        if (!display)

            display = av_mallocz(sizeof(DVBSubRegionDisplay));



        display->region_id = region_id;



        display->x_pos = AV_RB16(buf);

        buf += 2;

        display->y_pos = AV_RB16(buf);

        buf += 2;



        *tmp_ptr = display->next;



        display->next = ctx->display_list;

        ctx->display_list = display;

        ctx->display_list_size++;



        av_dlog(avctx, "Region %d, (%d,%d)\n", region_id, display->x_pos, display->y_pos);

    }



    while (tmp_display_list) {

        display = tmp_display_list;



        tmp_display_list = display->next;



        av_free(display);

    }



}
