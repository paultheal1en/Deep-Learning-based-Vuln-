static int pic_arrays_init(HEVCContext *s)

{

    int log2_min_cb_size     = s->sps->log2_min_cb_size;

    int width                = s->sps->width;

    int height               = s->sps->height;

    int pic_size             = width * height;

    int pic_size_in_ctb      = ((width  >> log2_min_cb_size) + 1) *

                               ((height >> log2_min_cb_size) + 1);

    int ctb_count            = s->sps->ctb_width * s->sps->ctb_height;

    int min_pu_width  = width  >> s->sps->log2_min_pu_size;

    int pic_height_in_min_pu = height >> s->sps->log2_min_pu_size;

    int pic_size_in_min_pu   = min_pu_width * pic_height_in_min_pu;

    int pic_width_in_min_tu  = width  >> s->sps->log2_min_tb_size;

    int pic_height_in_min_tu = height >> s->sps->log2_min_tb_size;



    s->bs_width  = width  >> 3;

    s->bs_height = height >> 3;



    s->sao           = av_mallocz_array(ctb_count, sizeof(*s->sao));

    s->deblock       = av_mallocz_array(ctb_count, sizeof(*s->deblock));

    s->split_cu_flag = av_malloc(pic_size);

    if (!s->sao || !s->deblock || !s->split_cu_flag)

        goto fail;



    s->skip_flag    = av_malloc(pic_size_in_ctb);

    s->tab_ct_depth = av_malloc(s->sps->min_cb_height * s->sps->min_cb_width);

    if (!s->skip_flag || !s->tab_ct_depth)

        goto fail;



    s->tab_ipm  = av_malloc(pic_size_in_min_pu);

    s->cbf_luma = av_malloc(pic_width_in_min_tu * pic_height_in_min_tu);

    s->is_pcm   = av_malloc(pic_size_in_min_pu);

    if (!s->tab_ipm || !s->cbf_luma || !s->is_pcm)

        goto fail;



    s->filter_slice_edges = av_malloc(ctb_count);

    s->tab_slice_address  = av_malloc(pic_size_in_ctb * sizeof(*s->tab_slice_address));

    s->qp_y_tab           = av_malloc(pic_size_in_ctb * sizeof(*s->qp_y_tab));

    if (!s->qp_y_tab || !s->filter_slice_edges || !s->tab_slice_address)

        goto fail;



    s->horizontal_bs = av_mallocz(2 * s->bs_width * (s->bs_height + 1));

    s->vertical_bs   = av_mallocz(2 * s->bs_width * (s->bs_height + 1));

    if (!s->horizontal_bs || !s->vertical_bs)

        goto fail;



    s->tab_mvf_pool = av_buffer_pool_init(pic_size_in_min_pu * sizeof(MvField),

                                          av_buffer_alloc);

    s->rpl_tab_pool = av_buffer_pool_init(ctb_count * sizeof(RefPicListTab),

                                          av_buffer_allocz);

    if (!s->tab_mvf_pool || !s->rpl_tab_pool)

        goto fail;



    return 0;

fail:

    pic_arrays_free(s);

    return AVERROR(ENOMEM);

}
