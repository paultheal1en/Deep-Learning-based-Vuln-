static int vc1_decode_frame(AVCodecContext *avctx, void *data,
                            int *data_size, AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size, n_slices = 0, i;
    VC1Context *v = avctx->priv_data;
    MpegEncContext *s = &v->s;
    AVFrame *pict = data;
    uint8_t *buf2 = NULL;
    const uint8_t *buf_start = buf, *buf_start_second_field = NULL;
    int mb_height, n_slices1=-1;
    struct {
        uint8_t *buf;
        GetBitContext gb;
        int mby_start;
    } *slices = NULL, *tmp;
    v->second_field = 0;
    if(s->flags & CODEC_FLAG_LOW_DELAY)
        s->low_delay = 1;
    /* no supplementary picture */
    if (buf_size == 0 || (buf_size == 4 && AV_RB32(buf) == VC1_CODE_ENDOFSEQ)) {
        /* special case for last picture */
        if (s->low_delay == 0 && s->next_picture_ptr) {
            *pict = s->next_picture_ptr->f;
            s->next_picture_ptr = NULL;
            *data_size = sizeof(AVFrame);
        return buf_size;
    if (s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU) {
        if (v->profile < PROFILE_ADVANCED)
            avctx->pix_fmt = AV_PIX_FMT_VDPAU_WMV3;
        else
            avctx->pix_fmt = AV_PIX_FMT_VDPAU_VC1;
    //for advanced profile we may need to parse and unescape data
    if (avctx->codec_id == AV_CODEC_ID_VC1 || avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {
        int buf_size2 = 0;
        buf2 = av_mallocz(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);
        if (IS_MARKER(AV_RB32(buf))) { /* frame starts with marker and needs to be parsed */
            const uint8_t *start, *end, *next;
            int size;
            next = buf;
            for (start = buf, end = buf + buf_size; next < end; start = next) {
                next = find_next_marker(start + 4, end);
                size = next - start - 4;
                if (size <= 0) continue;
                switch (AV_RB32(start)) {
                case VC1_CODE_FRAME:
                    if (avctx->hwaccel ||
                        s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU)
                        buf_start = start;
                    buf_size2 = vc1_unescape_buffer(start + 4, size, buf2);
                    break;
                case VC1_CODE_FIELD: {
                    int buf_size3;
                    if (avctx->hwaccel ||
                        s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU)
                        buf_start_second_field = start;
                    tmp = av_realloc(slices, sizeof(*slices) * (n_slices+1));
                    if (!tmp)
                    slices = tmp;
                    slices[n_slices].buf = av_mallocz(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);
                    if (!slices[n_slices].buf)
                    buf_size3 = vc1_unescape_buffer(start + 4, size,
                                                    slices[n_slices].buf);
                    init_get_bits(&slices[n_slices].gb, slices[n_slices].buf,
                                  buf_size3 << 3);
                    /* assuming that the field marker is at the exact middle,
                       hope it's correct */
                    slices[n_slices].mby_start = s->mb_height >> 1;
                    n_slices1 = n_slices - 1; // index of the last slice of the first field
                    n_slices++;
                    break;
                case VC1_CODE_ENTRYPOINT: /* it should be before frame data */
                    buf_size2 = vc1_unescape_buffer(start + 4, size, buf2);
                    init_get_bits(&s->gb, buf2, buf_size2 * 8);
                    ff_vc1_decode_entry_point(avctx, v, &s->gb);
                    break;
                case VC1_CODE_SLICE: {
                    int buf_size3;
                    tmp = av_realloc(slices, sizeof(*slices) * (n_slices+1));
                    if (!tmp)
                    slices = tmp;
                    slices[n_slices].buf = av_mallocz(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);
                    if (!slices[n_slices].buf)
                    buf_size3 = vc1_unescape_buffer(start + 4, size,
                                                    slices[n_slices].buf);
                    init_get_bits(&slices[n_slices].gb, slices[n_slices].buf,
                                  buf_size3 << 3);
                    slices[n_slices].mby_start = get_bits(&slices[n_slices].gb, 9);
                    n_slices++;
                    break;
        } else if (v->interlace && ((buf[0] & 0xC0) == 0xC0)) { /* WVC1 interlaced stores both fields divided by marker */
            const uint8_t *divider;
            int buf_size3;
            divider = find_next_marker(buf, buf + buf_size);
            if ((divider == (buf + buf_size)) || AV_RB32(divider) != VC1_CODE_FIELD) {
                av_log(avctx, AV_LOG_ERROR, "Error in WVC1 interlaced frame\n");
            } else { // found field marker, unescape second field
                if (avctx->hwaccel ||
                    s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU)
                    buf_start_second_field = divider;
                tmp = av_realloc(slices, sizeof(*slices) * (n_slices+1));
                if (!tmp)
                slices = tmp;
                slices[n_slices].buf = av_mallocz(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);
                if (!slices[n_slices].buf)
                buf_size3 = vc1_unescape_buffer(divider + 4, buf + buf_size - divider - 4, slices[n_slices].buf);
                init_get_bits(&slices[n_slices].gb, slices[n_slices].buf,
                              buf_size3 << 3);
                slices[n_slices].mby_start = s->mb_height >> 1;
                n_slices1 = n_slices - 1;
                n_slices++;
            buf_size2 = vc1_unescape_buffer(buf, divider - buf, buf2);
        } else {
            buf_size2 = vc1_unescape_buffer(buf, buf_size, buf2);
        init_get_bits(&s->gb, buf2, buf_size2*8);
    } else
        init_get_bits(&s->gb, buf, buf_size*8);
    if (v->res_sprite) {
        v->new_sprite  = !get_bits1(&s->gb);
        v->two_sprites =  get_bits1(&s->gb);
        /* res_sprite means a Windows Media Image stream, AV_CODEC_ID_*IMAGE means
           we're using the sprite compositor. These are intentionally kept separate
           so you can get the raw sprites by using the wmv3 decoder for WMVP or
           the vc1 one for WVP2 */
        if (avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {
            if (v->new_sprite) {
                // switch AVCodecContext parameters to those of the sprites
                avctx->width  = avctx->coded_width  = v->sprite_width;
                avctx->height = avctx->coded_height = v->sprite_height;
            } else {
                goto image;
    if (s->context_initialized &&
        (s->width  != avctx->coded_width ||
         s->height != avctx->coded_height)) {
        ff_vc1_decode_end(avctx);
    if (!s->context_initialized) {
        if (ff_msmpeg4_decode_init(avctx) < 0 || ff_vc1_decode_init_alloc_tables(v) < 0)
        s->low_delay = !avctx->has_b_frames || v->res_sprite;
        if (v->profile == PROFILE_ADVANCED) {
            s->h_edge_pos = avctx->coded_width;
            s->v_edge_pos = avctx->coded_height;
    /* We need to set current_picture_ptr before reading the header,
     * otherwise we cannot store anything in there. */
    if (s->current_picture_ptr == NULL || s->current_picture_ptr->f.data[0]) {
        int i = ff_find_unused_picture(s, 0);
        if (i < 0)
        s->current_picture_ptr = &s->picture[i];
    // do parse frame header
    v->pic_header_flag = 0;
    if (v->profile < PROFILE_ADVANCED) {
        if (ff_vc1_parse_frame_header(v, &s->gb) < 0) {
    } else {
        if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
    if (avctx->debug & FF_DEBUG_PICT_INFO)
        av_log(v->s.avctx, AV_LOG_DEBUG, "pict_type: %c\n", av_get_picture_type_char(s->pict_type));
    if ((avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE)
        && s->pict_type != AV_PICTURE_TYPE_I) {
        av_log(v->s.avctx, AV_LOG_ERROR, "Sprite decoder: expected I-frame\n");
    // process pulldown flags
    s->current_picture_ptr->f.repeat_pict = 0;
    // Pulldown flags are only valid when 'broadcast' has been set.
    // So ticks_per_frame will be 2
    if (v->rff) {
        // repeat field
        s->current_picture_ptr->f.repeat_pict = 1;
    } else if (v->rptfrm) {
        // repeat frames
        s->current_picture_ptr->f.repeat_pict = v->rptfrm * 2;
    // for skipping the frame
    s->current_picture.f.pict_type = s->pict_type;
    s->current_picture.f.key_frame = s->pict_type == AV_PICTURE_TYPE_I;
    /* skip B-frames if we don't have reference frames */
    if (s->last_picture_ptr == NULL && (s->pict_type == AV_PICTURE_TYPE_B || s->dropable)) {
    if ((avctx->skip_frame >= AVDISCARD_NONREF && s->pict_type == AV_PICTURE_TYPE_B) ||
        (avctx->skip_frame >= AVDISCARD_NONKEY && s->pict_type != AV_PICTURE_TYPE_I) ||
         avctx->skip_frame >= AVDISCARD_ALL) {
        goto end;
    if (s->next_p_frame_damaged) {
        if (s->pict_type == AV_PICTURE_TYPE_B)
            goto end;
        else
            s->next_p_frame_damaged = 0;
    if (ff_MPV_frame_start(s, avctx) < 0) {
    v->s.current_picture_ptr->f.interlaced_frame = (v->fcm != PROGRESSIVE);
    v->s.current_picture_ptr->f.top_field_first  = v->tff;
    s->me.qpel_put = s->dsp.put_qpel_pixels_tab;
    s->me.qpel_avg = s->dsp.avg_qpel_pixels_tab;
    if ((CONFIG_VC1_VDPAU_DECODER)
        &&s->avctx->codec->capabilities&CODEC_CAP_HWACCEL_VDPAU)
        ff_vdpau_vc1_decode_picture(s, buf_start, (buf + buf_size) - buf_start);
    else if (avctx->hwaccel) {
        if (v->field_mode && buf_start_second_field) {
            // decode first field
            s->picture_structure = PICT_BOTTOM_FIELD - v->tff;
            if (avctx->hwaccel->start_frame(avctx, buf_start, buf_start_second_field - buf_start) < 0)
            if (avctx->hwaccel->decode_slice(avctx, buf_start, buf_start_second_field - buf_start) < 0)
            if (avctx->hwaccel->end_frame(avctx) < 0)
            // decode second field
            s->gb = slices[n_slices1 + 1].gb;
            s->picture_structure = PICT_TOP_FIELD + v->tff;
            v->second_field = 1;
            v->pic_header_flag = 0;
            if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                av_log(avctx, AV_LOG_ERROR, "parsing header for second field failed");
            v->s.current_picture_ptr->f.pict_type = v->s.pict_type;
            if (avctx->hwaccel->start_frame(avctx, buf_start_second_field, (buf + buf_size) - buf_start_second_field) < 0)
            if (avctx->hwaccel->decode_slice(avctx, buf_start_second_field, (buf + buf_size) - buf_start_second_field) < 0)
            if (avctx->hwaccel->end_frame(avctx) < 0)
        } else {
            s->picture_structure = PICT_FRAME;
            if (avctx->hwaccel->start_frame(avctx, buf_start, (buf + buf_size) - buf_start) < 0)
            if (avctx->hwaccel->decode_slice(avctx, buf_start, (buf + buf_size) - buf_start) < 0)
            if (avctx->hwaccel->end_frame(avctx) < 0)
    } else {
        if (v->fcm == ILACE_FRAME && s->pict_type == AV_PICTURE_TYPE_B)
            goto err; // This codepath is still incomplete thus it is disabled
        ff_er_frame_start(s);
        v->bits = buf_size * 8;
        v->end_mb_x = s->mb_width;
        if (v->field_mode) {
            uint8_t *tmp[2];
            s->current_picture.f.linesize[0] <<= 1;
            s->current_picture.f.linesize[1] <<= 1;
            s->current_picture.f.linesize[2] <<= 1;
            s->linesize                      <<= 1;
            s->uvlinesize                    <<= 1;
            tmp[0]          = v->mv_f_last[0];
            tmp[1]          = v->mv_f_last[1];
            v->mv_f_last[0] = v->mv_f_next[0];
            v->mv_f_last[1] = v->mv_f_next[1];
            v->mv_f_next[0] = v->mv_f[0];
            v->mv_f_next[1] = v->mv_f[1];
            v->mv_f[0] = tmp[0];
            v->mv_f[1] = tmp[1];
        mb_height = s->mb_height >> v->field_mode;
        for (i = 0; i <= n_slices; i++) {
            if (i > 0 &&  slices[i - 1].mby_start >= mb_height) {
                if (v->field_mode <= 0) {
                    av_log(v->s.avctx, AV_LOG_ERROR, "Slice %d starts beyond "
                           "picture boundary (%d >= %d)\n", i,
                           slices[i - 1].mby_start, mb_height);
                    continue;
                v->second_field = 1;
                v->blocks_off   = s->mb_width  * s->mb_height << 1;
                v->mb_off       = s->mb_stride * s->mb_height >> 1;
            } else {
                v->second_field = 0;
                v->blocks_off   = 0;
                v->mb_off       = 0;
            if (i) {
                v->pic_header_flag = 0;
                if (v->field_mode && i == n_slices1 + 2) {
                    if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                        av_log(v->s.avctx, AV_LOG_ERROR, "Field header damaged\n");
                        continue;
                } else if (get_bits1(&s->gb)) {
                    v->pic_header_flag = 1;
                    if (ff_vc1_parse_frame_header_adv(v, &s->gb) < 0) {
                        av_log(v->s.avctx, AV_LOG_ERROR, "Slice header damaged\n");
                        continue;
            s->start_mb_y = (i == 0) ? 0 : FFMAX(0, slices[i-1].mby_start % mb_height);
            if (!v->field_mode || v->second_field)
                s->end_mb_y = (i == n_slices     ) ? mb_height : FFMIN(mb_height, slices[i].mby_start % mb_height);
            else
                s->end_mb_y = (i <= n_slices1 + 1) ? mb_height : FFMIN(mb_height, slices[i].mby_start % mb_height);
            if (s->end_mb_y <= s->start_mb_y) {
                av_log(v->s.avctx, AV_LOG_ERROR, "end mb y %d %d invalid\n", s->end_mb_y, s->start_mb_y);
                continue;
            ff_vc1_decode_blocks(v);
            if (i != n_slices)
                s->gb = slices[i].gb;
        if (v->field_mode) {
            v->second_field = 0;
            if (s->pict_type == AV_PICTURE_TYPE_B) {
                memcpy(v->mv_f_base, v->mv_f_next_base,
                       2 * (s->b8_stride * (s->mb_height * 2 + 1) + s->mb_stride * (s->mb_height + 1) * 2));
            s->current_picture.f.linesize[0] >>= 1;
            s->current_picture.f.linesize[1] >>= 1;
            s->current_picture.f.linesize[2] >>= 1;
            s->linesize                      >>= 1;
            s->uvlinesize                    >>= 1;
        av_dlog(s->avctx, "Consumed %i/%i bits\n",
                get_bits_count(&s->gb), s->gb.size_in_bits);
//  if (get_bits_count(&s->gb) > buf_size * 8)
//      return -1;
        if(s->error_occurred && s->pict_type == AV_PICTURE_TYPE_B)
        if(!v->field_mode)
            ff_er_frame_end(s);
    ff_MPV_frame_end(s);
    if (avctx->codec_id == AV_CODEC_ID_WMV3IMAGE || avctx->codec_id == AV_CODEC_ID_VC1IMAGE) {
image:
        avctx->width  = avctx->coded_width  = v->output_width;
        avctx->height = avctx->coded_height = v->output_height;
        if (avctx->skip_frame >= AVDISCARD_NONREF)
            goto end;
#if CONFIG_WMV3IMAGE_DECODER || CONFIG_VC1IMAGE_DECODER
        if (vc1_decode_sprites(v, &s->gb))
#endif
        *pict      = v->sprite_output_frame;
        *data_size = sizeof(AVFrame);
    } else {
        if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {
            *pict = s->current_picture_ptr->f;
        } else if (s->last_picture_ptr != NULL) {
            *pict = s->last_picture_ptr->f;
        if (s->last_picture_ptr || s->low_delay) {
            *data_size = sizeof(AVFrame);
            ff_print_debug_info(s, pict);
end:
    av_free(buf2);
    for (i = 0; i < n_slices; i++)
        av_free(slices[i].buf);
    av_free(slices);
    return buf_size;
err:
    av_free(buf2);
    for (i = 0; i < n_slices; i++)
        av_free(slices[i].buf);
    av_free(slices);
    return -1;