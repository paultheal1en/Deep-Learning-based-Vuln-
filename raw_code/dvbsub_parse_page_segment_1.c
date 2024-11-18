static int dvbsub_parse_page_segment(AVCodecContext *avctx,
                                     const uint8_t *buf, int buf_size, AVSubtitle *sub, int *got_output)
{
    DVBSubContext *ctx = avctx->priv_data;
    DVBSubRegionDisplay *display;
    DVBSubRegionDisplay *tmp_display_list, **tmp_ptr;
    const uint8_t *buf_end = buf + buf_size;
    int region_id;
    int page_state;
    int timeout;
    int version;
    if (buf_size < 1)
        return AVERROR_INVALIDDATA;
    timeout = *buf++;
    version = ((*buf)>>4) & 15;
    page_state = ((*buf++) >> 2) & 3;
    if (ctx->version == version) {
        return 0;
    ctx->time_out = timeout;
    ctx->version = version;
    ff_dlog(avctx, "Page time out %ds, state %d\n", ctx->time_out, page_state);
    if(ctx->compute_edt == 1)
        save_subtitle_set(avctx, sub, got_output);
    if (page_state == 1 || page_state == 2) {
        delete_regions(ctx);
        delete_objects(ctx);
        delete_cluts(ctx);
    tmp_display_list = ctx->display_list;
    ctx->display_list = NULL;
    while (buf + 5 < buf_end) {
        region_id = *buf++;
        buf += 1;
        display = tmp_display_list;
        tmp_ptr = &tmp_display_list;
            tmp_ptr = &display->next;
        if (!display) {
            display = av_mallocz(sizeof(DVBSubRegionDisplay));
            if (!display)
                return AVERROR(ENOMEM);
        display->region_id = region_id;
        display->x_pos = AV_RB16(buf);
        buf += 2;
        display->y_pos = AV_RB16(buf);
        buf += 2;
        *tmp_ptr = display->next;
        display->next = ctx->display_list;
        ctx->display_list = display;
        ff_dlog(avctx, "Region %d, (%d,%d)\n", region_id, display->x_pos, display->y_pos);
    while (tmp_display_list) {
        display = tmp_display_list;
        tmp_display_list = display->next;
        av_freep(&display);
    return 0;