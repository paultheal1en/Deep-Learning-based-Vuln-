static int dirac_unpack_prediction_parameters(DiracContext *s)

{

    static const uint8_t default_blen[] = { 4, 12, 16, 24 };

    static const uint8_t default_bsep[] = { 4,  8, 12, 16 };



    GetBitContext *gb = &s->gb;

    unsigned idx, ref;



    align_get_bits(gb);

    /* [DIRAC_STD] 11.2.2 Block parameters. block_parameters() */

    /* Luma and Chroma are equal. 11.2.3 */

    idx = svq3_get_ue_golomb(gb); /* [DIRAC_STD] index */



    if (idx > 4) {

        av_log(s->avctx, AV_LOG_ERROR, "Block prediction index too high\n");

        return -1;

    }



    if (idx == 0) {

        s->plane[0].xblen = svq3_get_ue_golomb(gb);

        s->plane[0].yblen = svq3_get_ue_golomb(gb);

        s->plane[0].xbsep = svq3_get_ue_golomb(gb);

        s->plane[0].ybsep = svq3_get_ue_golomb(gb);

    } else {

        /*[DIRAC_STD] preset_block_params(index). Table 11.1 */

        s->plane[0].xblen = default_blen[idx-1];

        s->plane[0].yblen = default_blen[idx-1];

        s->plane[0].xbsep = default_bsep[idx-1];

        s->plane[0].ybsep = default_bsep[idx-1];

    }

    /*[DIRAC_STD] 11.2.4 motion_data_dimensions()

      Calculated in function dirac_unpack_block_motion_data */



    if (s->plane[0].xbsep < s->plane[0].xblen/2 || s->plane[0].ybsep < s->plane[0].yblen/2) {

        av_log(s->avctx, AV_LOG_ERROR, "Block separation too small\n");

        return -1;

    }

    if (s->plane[0].xbsep > s->plane[0].xblen || s->plane[0].ybsep > s->plane[0].yblen) {

        av_log(s->avctx, AV_LOG_ERROR, "Block seperation greater than size\n");

        return -1;

    }

    if (FFMAX(s->plane[0].xblen, s->plane[0].yblen) > MAX_BLOCKSIZE) {

        av_log(s->avctx, AV_LOG_ERROR, "Unsupported large block size\n");

        return -1;

    }



    /*[DIRAC_STD] 11.2.5 Motion vector precision. motion_vector_precision()

      Read motion vector precision */

    s->mv_precision = svq3_get_ue_golomb(gb);

    if (s->mv_precision > 3) {

        av_log(s->avctx, AV_LOG_ERROR, "MV precision finer than eighth-pel\n");

        return -1;

    }



    /*[DIRAC_STD] 11.2.6 Global motion. global_motion()

      Read the global motion compensation parameters */

    s->globalmc_flag = get_bits1(gb);

    if (s->globalmc_flag) {

        memset(s->globalmc, 0, sizeof(s->globalmc));

        /* [DIRAC_STD] pan_tilt(gparams) */

        for (ref = 0; ref < s->num_refs; ref++) {

            if (get_bits1(gb)) {

                s->globalmc[ref].pan_tilt[0] = dirac_get_se_golomb(gb);

                s->globalmc[ref].pan_tilt[1] = dirac_get_se_golomb(gb);

            }

            /* [DIRAC_STD] zoom_rotate_shear(gparams)

               zoom/rotation/shear parameters */

            if (get_bits1(gb)) {

                s->globalmc[ref].zrs_exp   = svq3_get_ue_golomb(gb);

                s->globalmc[ref].zrs[0][0] = dirac_get_se_golomb(gb);

                s->globalmc[ref].zrs[0][1] = dirac_get_se_golomb(gb);

                s->globalmc[ref].zrs[1][0] = dirac_get_se_golomb(gb);

                s->globalmc[ref].zrs[1][1] = dirac_get_se_golomb(gb);

            } else {

                s->globalmc[ref].zrs[0][0] = 1;

                s->globalmc[ref].zrs[1][1] = 1;

            }

            /* [DIRAC_STD] perspective(gparams) */

            if (get_bits1(gb)) {

                s->globalmc[ref].perspective_exp = svq3_get_ue_golomb(gb);

                s->globalmc[ref].perspective[0]  = dirac_get_se_golomb(gb);

                s->globalmc[ref].perspective[1]  = dirac_get_se_golomb(gb);

            }

        }

    }



    /*[DIRAC_STD] 11.2.7 Picture prediction mode. prediction_mode()

      Picture prediction mode, not currently used. */

    if (svq3_get_ue_golomb(gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "Unknown picture prediction mode\n");

        return -1;

    }



    /* [DIRAC_STD] 11.2.8 Reference picture weight. reference_picture_weights()

       just data read, weight calculation will be done later on. */

    s->weight_log2denom = 1;

    s->weight[0]        = 1;

    s->weight[1]        = 1;



    if (get_bits1(gb)) {

        s->weight_log2denom = svq3_get_ue_golomb(gb);

        s->weight[0] = dirac_get_se_golomb(gb);

        if (s->num_refs == 2)

            s->weight[1] = dirac_get_se_golomb(gb);

    }

    return 0;

}
