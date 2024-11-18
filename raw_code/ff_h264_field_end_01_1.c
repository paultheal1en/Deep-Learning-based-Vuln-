int ff_h264_field_end(H264Context *h, int in_setup)
{
    AVCodecContext *const avctx = h->avctx;
    int err = 0;
    h->mb_y = 0;
    if (CONFIG_H264_VDPAU_DECODER &&
        h->avctx->codec->capabilities & CODEC_CAP_HWACCEL_VDPAU)
        ff_vdpau_h264_set_reference_frames(h);
    if (in_setup || !(avctx->active_thread_type & FF_THREAD_FRAME)) {
        if (!h->droppable) {
            err = ff_h264_execute_ref_pic_marking(h, h->mmco, h->mmco_index);
            h->prev_poc_msb = h->poc_msb;
            h->prev_poc_lsb = h->poc_lsb;
        }
        h->prev_frame_num_offset = h->frame_num_offset;
        h->prev_frame_num        = h->frame_num;
        h->outputed_poc          = h->next_outputed_poc;
    }
    if (avctx->hwaccel) {
        if (avctx->hwaccel->end_frame(avctx) < 0)
            av_log(avctx, AV_LOG_ERROR,
                   "hardware accelerator failed to decode picture\n");
    }
    if (CONFIG_H264_VDPAU_DECODER &&
        h->avctx->codec->capabilities & CODEC_CAP_HWACCEL_VDPAU)
        ff_vdpau_h264_picture_complete(h);
#if CONFIG_ERROR_RESILIENCE
    /*
     * FIXME: Error handling code does not seem to support interlaced
     * when slices span multiple rows
     * The ff_er_add_slice calls don't work right for bottom
     * fields; they cause massive erroneous error concealing
     * Error marking covers both fields (top and bottom).
     * This causes a mismatched s->error_count
     * and a bad error table. Further, the error count goes to
     * INT_MAX when called for bottom field, because mb_y is
     * past end by one (callers fault) and resync_mb_y != 0
     * causes problems for the first MB line, too.
     */
    if (!FIELD_PICTURE(h) && h->current_slice && !h->sps.new) {
        ff_h264_set_erpic(&h->er.cur_pic, h->cur_pic_ptr);
        ff_er_frame_end(&h->er);
    }
#endif /* CONFIG_ERROR_RESILIENCE */
    if (!in_setup && !h->droppable)
        ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,
                                  h->picture_structure == PICT_BOTTOM_FIELD);
    emms_c();
    h->current_slice = 0;
    return err;
}