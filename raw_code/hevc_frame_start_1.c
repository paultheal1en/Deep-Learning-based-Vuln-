static int hevc_frame_start(HEVCContext *s)

{

    HEVCLocalContext *lc = &s->HEVClc;

    int ret;



    memset(s->horizontal_bs, 0, 2 * s->bs_width * (s->bs_height + 1));

    memset(s->vertical_bs,   0, 2 * s->bs_width * (s->bs_height + 1));

    memset(s->cbf_luma,      0, s->sps->min_tb_width * s->sps->min_tb_height);

    memset(s->is_pcm,        0, s->sps->min_pu_width * s->sps->min_pu_height);



    lc->start_of_tiles_x = 0;

    s->is_decoded        = 0;




    if (s->pps->tiles_enabled_flag)

        lc->end_of_tiles_x = s->pps->column_width[0] << s->sps->log2_ctb_size;



    ret = ff_hevc_set_new_ref(s, s->sps->sao_enabled ? &s->sao_frame : &s->frame,

                              s->poc);

    if (ret < 0)

        goto fail;



    ret = ff_hevc_frame_rps(s);

    if (ret < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "Error constructing the frame RPS.\n");

        goto fail;

    }



    ret = set_side_data(s);

    if (ret < 0)

        goto fail;



    av_frame_unref(s->output_frame);

    ret = ff_hevc_output_frame(s, s->output_frame, 0);

    if (ret < 0)

        goto fail;



    ff_thread_finish_setup(s->avctx);



    return 0;



fail:

    if (s->ref)

        ff_thread_report_progress(&s->ref->tf, INT_MAX, 0);

    s->ref = NULL;

    return ret;

}