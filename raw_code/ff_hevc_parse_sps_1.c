int ff_hevc_parse_sps(HEVCSPS *sps, GetBitContext *gb, unsigned int *sps_id,

                      int apply_defdispwin, AVBufferRef **vps_list, AVCodecContext *avctx)

{

    HEVCWindow *ow;

    int ret = 0;

    int log2_diff_max_min_transform_block_size;

    int bit_depth_chroma, start, vui_present, sublayer_ordering_info;

    int i;



    // Coded parameters



    sps->vps_id = get_bits(gb, 4);

    if (sps->vps_id >= HEVC_MAX_VPS_COUNT) {

        av_log(avctx, AV_LOG_ERROR, "VPS id out of range: %d\n", sps->vps_id);






    if (vps_list && !vps_list[sps->vps_id]) {

        av_log(avctx, AV_LOG_ERROR, "VPS %d does not exist\n",

               sps->vps_id);






    sps->max_sub_layers = get_bits(gb, 3) + 1;

    if (sps->max_sub_layers > HEVC_MAX_SUB_LAYERS) {

        av_log(avctx, AV_LOG_ERROR, "sps_max_sub_layers out of range: %d\n",

               sps->max_sub_layers);






    skip_bits1(gb); // temporal_id_nesting_flag



    parse_ptl(gb, avctx, &sps->ptl, sps->max_sub_layers);



    *sps_id = get_ue_golomb_long(gb);

    if (*sps_id >= HEVC_MAX_SPS_COUNT) {

        av_log(avctx, AV_LOG_ERROR, "SPS id out of range: %d\n", *sps_id);






    sps->chroma_format_idc = get_ue_golomb_long(gb);

    if (sps->chroma_format_idc != 1) {

        avpriv_report_missing_feature(avctx, "chroma_format_idc %d",

                                      sps->chroma_format_idc);

        ret = AVERROR_PATCHWELCOME;





    if (sps->chroma_format_idc == 3)

        sps->separate_colour_plane_flag = get_bits1(gb);



    sps->width  = get_ue_golomb_long(gb);

    sps->height = get_ue_golomb_long(gb);

    if ((ret = av_image_check_size(sps->width,

                                   sps->height, 0, avctx)) < 0)




    if (get_bits1(gb)) { // pic_conformance_flag

        //TODO: * 2 is only valid for 420

        sps->pic_conf_win.left_offset   = get_ue_golomb_long(gb) * 2;

        sps->pic_conf_win.right_offset  = get_ue_golomb_long(gb) * 2;

        sps->pic_conf_win.top_offset    = get_ue_golomb_long(gb) * 2;

        sps->pic_conf_win.bottom_offset = get_ue_golomb_long(gb) * 2;



        if (avctx->flags2 & AV_CODEC_FLAG2_IGNORE_CROP) {

            av_log(avctx, AV_LOG_DEBUG,

                   "discarding sps conformance window, "

                   "original values are l:%u r:%u t:%u b:%u\n",

                   sps->pic_conf_win.left_offset,

                   sps->pic_conf_win.right_offset,

                   sps->pic_conf_win.top_offset,

                   sps->pic_conf_win.bottom_offset);



            sps->pic_conf_win.left_offset   =

            sps->pic_conf_win.right_offset  =

            sps->pic_conf_win.top_offset    =

            sps->pic_conf_win.bottom_offset = 0;


        sps->output_window = sps->pic_conf_win;




    sps->bit_depth   = get_ue_golomb_long(gb) + 8;

    bit_depth_chroma = get_ue_golomb_long(gb) + 8;

    if (bit_depth_chroma != sps->bit_depth) {

        av_log(avctx, AV_LOG_ERROR,

               "Luma bit depth (%d) is different from chroma bit depth (%d), "

               "this is unsupported.\n",

               sps->bit_depth, bit_depth_chroma);








    ret = map_pixel_format(avctx, sps);

    if (ret < 0)




    sps->log2_max_poc_lsb = get_ue_golomb_long(gb) + 4;

    if (sps->log2_max_poc_lsb > 16) {

        av_log(avctx, AV_LOG_ERROR, "log2_max_pic_order_cnt_lsb_minus4 out range: %d\n",

               sps->log2_max_poc_lsb - 4);






    sublayer_ordering_info = get_bits1(gb);

    start = sublayer_ordering_info ? 0 : sps->max_sub_layers - 1;

    for (i = start; i < sps->max_sub_layers; i++) {

        sps->temporal_layer[i].max_dec_pic_buffering = get_ue_golomb_long(gb) + 1;

        sps->temporal_layer[i].num_reorder_pics      = get_ue_golomb_long(gb);

        sps->temporal_layer[i].max_latency_increase  = get_ue_golomb_long(gb) - 1;

        if (sps->temporal_layer[i].max_dec_pic_buffering > HEVC_MAX_DPB_SIZE) {

            av_log(avctx, AV_LOG_ERROR, "sps_max_dec_pic_buffering_minus1 out of range: %d\n",

                   sps->temporal_layer[i].max_dec_pic_buffering - 1);




        if (sps->temporal_layer[i].num_reorder_pics > sps->temporal_layer[i].max_dec_pic_buffering - 1) {

            av_log(avctx, AV_LOG_WARNING, "sps_max_num_reorder_pics out of range: %d\n",

                   sps->temporal_layer[i].num_reorder_pics);

            if (avctx->err_recognition & AV_EF_EXPLODE ||

                sps->temporal_layer[i].num_reorder_pics > HEVC_MAX_DPB_SIZE - 1) {




            sps->temporal_layer[i].max_dec_pic_buffering = sps->temporal_layer[i].num_reorder_pics + 1;





    if (!sublayer_ordering_info) {

        for (i = 0; i < start; i++) {

            sps->temporal_layer[i].max_dec_pic_buffering = sps->temporal_layer[start].max_dec_pic_buffering;

            sps->temporal_layer[i].num_reorder_pics      = sps->temporal_layer[start].num_reorder_pics;

            sps->temporal_layer[i].max_latency_increase  = sps->temporal_layer[start].max_latency_increase;





    sps->log2_min_cb_size                    = get_ue_golomb_long(gb) + 3;

    sps->log2_diff_max_min_coding_block_size = get_ue_golomb_long(gb);

    sps->log2_min_tb_size                    = get_ue_golomb_long(gb) + 2;

    log2_diff_max_min_transform_block_size   = get_ue_golomb_long(gb);

    sps->log2_max_trafo_size                 = log2_diff_max_min_transform_block_size +

                                               sps->log2_min_tb_size;



    if (sps->log2_min_tb_size >= sps->log2_min_cb_size) {

        av_log(avctx, AV_LOG_ERROR, "Invalid value for log2_min_tb_size");




    sps->max_transform_hierarchy_depth_inter = get_ue_golomb_long(gb);

    sps->max_transform_hierarchy_depth_intra = get_ue_golomb_long(gb);



    sps->scaling_list_enable_flag = get_bits1(gb);

    if (sps->scaling_list_enable_flag) {

        set_default_scaling_list_data(&sps->scaling_list);



        if (get_bits1(gb)) {

            ret = scaling_list_data(gb, avctx, &sps->scaling_list);

            if (ret < 0)






    sps->amp_enabled_flag = get_bits1(gb);

    sps->sao_enabled      = get_bits1(gb);



    sps->pcm_enabled_flag = get_bits1(gb);

    if (sps->pcm_enabled_flag) {

        sps->pcm.bit_depth   = get_bits(gb, 4) + 1;

        sps->pcm.bit_depth_chroma = get_bits(gb, 4) + 1;

        sps->pcm.log2_min_pcm_cb_size = get_ue_golomb_long(gb) + 3;

        sps->pcm.log2_max_pcm_cb_size = sps->pcm.log2_min_pcm_cb_size +

                                        get_ue_golomb_long(gb);

        if (sps->pcm.bit_depth > sps->bit_depth) {

            av_log(avctx, AV_LOG_ERROR,

                   "PCM bit depth (%d) is greater than normal bit depth (%d)\n",

                   sps->pcm.bit_depth, sps->bit_depth);






        sps->pcm.loop_filter_disable_flag = get_bits1(gb);




    sps->nb_st_rps = get_ue_golomb_long(gb);

    if (sps->nb_st_rps > HEVC_MAX_SHORT_TERM_REF_PIC_SETS) {

        av_log(avctx, AV_LOG_ERROR, "Too many short term RPS: %d.\n",

               sps->nb_st_rps);




    for (i = 0; i < sps->nb_st_rps; i++) {

        if ((ret = ff_hevc_decode_short_term_rps(gb, avctx, &sps->st_rps[i],

                                                 sps, 0)) < 0)





    sps->long_term_ref_pics_present_flag = get_bits1(gb);

    if (sps->long_term_ref_pics_present_flag) {

        sps->num_long_term_ref_pics_sps = get_ue_golomb_long(gb);







        for (i = 0; i < sps->num_long_term_ref_pics_sps; i++) {

            sps->lt_ref_pic_poc_lsb_sps[i]       = get_bits(gb, sps->log2_max_poc_lsb);

            sps->used_by_curr_pic_lt_sps_flag[i] = get_bits1(gb);





    sps->sps_temporal_mvp_enabled_flag          = get_bits1(gb);

    sps->sps_strong_intra_smoothing_enable_flag = get_bits1(gb);

    sps->vui.sar = (AVRational){0, 1};

    vui_present = get_bits1(gb);

    if (vui_present)

        decode_vui(gb, avctx, apply_defdispwin, sps);

    skip_bits1(gb); // sps_extension_flag



    if (apply_defdispwin) {

        sps->output_window.left_offset   += sps->vui.def_disp_win.left_offset;

        sps->output_window.right_offset  += sps->vui.def_disp_win.right_offset;

        sps->output_window.top_offset    += sps->vui.def_disp_win.top_offset;

        sps->output_window.bottom_offset += sps->vui.def_disp_win.bottom_offset;




    ow = &sps->output_window;

    if (ow->left_offset >= INT_MAX - ow->right_offset     ||

        ow->top_offset  >= INT_MAX - ow->bottom_offset    ||

        ow->left_offset + ow->right_offset  >= sps->width ||

        ow->top_offset  + ow->bottom_offset >= sps->height) {

        av_log(avctx, AV_LOG_WARNING, "Invalid cropping offsets: %u/%u/%u/%u\n",

               ow->left_offset, ow->right_offset, ow->top_offset, ow->bottom_offset);

        if (avctx->err_recognition & AV_EF_EXPLODE) {




        av_log(avctx, AV_LOG_WARNING,

               "Displaying the whole video surface.\n");

        memset(ow, 0, sizeof(*ow));




    // Inferred parameters

    sps->log2_ctb_size = sps->log2_min_cb_size +

                         sps->log2_diff_max_min_coding_block_size;

    sps->log2_min_pu_size = sps->log2_min_cb_size - 1;



    sps->ctb_width  = (sps->width  + (1 << sps->log2_ctb_size) - 1) >> sps->log2_ctb_size;

    sps->ctb_height = (sps->height + (1 << sps->log2_ctb_size) - 1) >> sps->log2_ctb_size;

    sps->ctb_size   = sps->ctb_width * sps->ctb_height;



    sps->min_cb_width  = sps->width  >> sps->log2_min_cb_size;

    sps->min_cb_height = sps->height >> sps->log2_min_cb_size;

    sps->min_tb_width  = sps->width  >> sps->log2_min_tb_size;

    sps->min_tb_height = sps->height >> sps->log2_min_tb_size;

    sps->min_pu_width  = sps->width  >> sps->log2_min_pu_size;

    sps->min_pu_height = sps->height >> sps->log2_min_pu_size;



    sps->qp_bd_offset = 6 * (sps->bit_depth - 8);



    if (sps->width  & ((1 << sps->log2_min_cb_size) - 1) ||

        sps->height & ((1 << sps->log2_min_cb_size) - 1)) {

        av_log(avctx, AV_LOG_ERROR, "Invalid coded frame dimensions.\n");





    if (sps->log2_ctb_size > HEVC_MAX_LOG2_CTB_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "CTB size out of range: 2^%d\n", sps->log2_ctb_size);



    if (sps->max_transform_hierarchy_depth_inter > sps->log2_ctb_size - sps->log2_min_tb_size) {

        av_log(avctx, AV_LOG_ERROR, "max_transform_hierarchy_depth_inter out of range: %d\n",

               sps->max_transform_hierarchy_depth_inter);



    if (sps->max_transform_hierarchy_depth_intra > sps->log2_ctb_size - sps->log2_min_tb_size) {

        av_log(avctx, AV_LOG_ERROR, "max_transform_hierarchy_depth_intra out of range: %d\n",

               sps->max_transform_hierarchy_depth_intra);



    if (sps->log2_max_trafo_size > FFMIN(sps->log2_ctb_size, 5)) {

        av_log(avctx, AV_LOG_ERROR,

               "max transform block size out of range: %d\n",

               sps->log2_max_trafo_size);





    return 0;



err:

    return ret < 0 ? ret : AVERROR_INVALIDDATA;
