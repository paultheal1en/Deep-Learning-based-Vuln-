static int cmv_process_header(CmvContext *s, const uint8_t *buf, const uint8_t *buf_end)
{
    int pal_start, pal_count, i, ret, fps;
    if(buf_end - buf < 16) {
        av_log(s->avctx, AV_LOG_WARNING, "truncated header\n");
        return AVERROR_INVALIDDATA;
    s->width  = AV_RL16(&buf[4]);
    s->height = AV_RL16(&buf[6]);
    ret = ff_set_dimensions(s->avctx, s->width, s->height);
    if (ret < 0)
        return ret;
    fps = AV_RL16(&buf[10]);
    if (fps > 0)
        s->avctx->time_base = (AVRational){ 1, fps };
    pal_start = AV_RL16(&buf[12]);
    pal_count = AV_RL16(&buf[14]);
    buf += 16;
    for (i=pal_start; i<pal_start+pal_count && i<AVPALETTE_COUNT && buf_end - buf >= 3; i++) {
        s->palette[i] = AV_RB24(buf);
        buf += 3;
    return 0;