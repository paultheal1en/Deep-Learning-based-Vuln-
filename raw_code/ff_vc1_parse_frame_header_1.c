int ff_vc1_parse_frame_header(VC1Context *v, GetBitContext* gb)

{

    int pqindex, lowquant, status;





    if (v->finterpflag)

        v->interpfrm = get_bits1(gb);

    if (!v->s.avctx->codec)

        return -1;

    if (v->s.avctx->codec_id == AV_CODEC_ID_MSS2)

        v->respic   =

        v->rangered =

        v->multires = get_bits(gb, 2) == 1;

    else

        skip_bits(gb, 2); //framecnt unused

    v->rangeredfrm = 0;

    if (v->rangered)

        v->rangeredfrm = get_bits1(gb);

    v->s.pict_type = get_bits1(gb);

    if (v->s.avctx->max_b_frames) {

        if (!v->s.pict_type) {

            if (get_bits1(gb))

                v->s.pict_type = AV_PICTURE_TYPE_I;

            else

                v->s.pict_type = AV_PICTURE_TYPE_B;

        } else

            v->s.pict_type = AV_PICTURE_TYPE_P;

    } else

        v->s.pict_type = v->s.pict_type ? AV_PICTURE_TYPE_P : AV_PICTURE_TYPE_I;



    v->bi_type = 0;

    if (v->s.pict_type == AV_PICTURE_TYPE_B) {

        if (read_bfraction(v, gb) < 0)

            return AVERROR_INVALIDDATA;

        if (v->bfraction == 0) {

            v->s.pict_type = AV_PICTURE_TYPE_BI;

        }

    }

    if (v->s.pict_type == AV_PICTURE_TYPE_I || v->s.pict_type == AV_PICTURE_TYPE_BI)

        skip_bits(gb, 7); // skip buffer fullness



    if (v->parse_only)

        return 0;



    /* calculate RND */

    if (v->s.pict_type == AV_PICTURE_TYPE_I || v->s.pict_type == AV_PICTURE_TYPE_BI)

        v->rnd = 1;

    if (v->s.pict_type == AV_PICTURE_TYPE_P)

        v->rnd ^= 1;



    /* Quantizer stuff */

    pqindex = get_bits(gb, 5);

    if (!pqindex)

        return -1;

    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)

        v->pq = ff_vc1_pquant_table[0][pqindex];

    else

        v->pq = ff_vc1_pquant_table[1][pqindex];



    v->pquantizer = 1;

    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)

        v->pquantizer = pqindex < 9;

    if (v->quantizer_mode == QUANT_NON_UNIFORM)

        v->pquantizer = 0;

    v->pqindex = pqindex;

    if (pqindex < 9)

        v->halfpq = get_bits1(gb);

    else

        v->halfpq = 0;

    if (v->quantizer_mode == QUANT_FRAME_EXPLICIT)

        v->pquantizer = get_bits1(gb);

    v->dquantfrm = 0;

    if (v->extended_mv == 1)

        v->mvrange = get_unary(gb, 0, 3);

    v->k_x = v->mvrange + 9 + (v->mvrange >> 1); //k_x can be 9 10 12 13

    v->k_y = v->mvrange + 8; //k_y can be 8 9 10 11

    v->range_x = 1 << (v->k_x - 1);

    v->range_y = 1 << (v->k_y - 1);

    if (v->multires && v->s.pict_type != AV_PICTURE_TYPE_B)

        v->respic = get_bits(gb, 2);



    if (v->res_x8 && (v->s.pict_type == AV_PICTURE_TYPE_I || v->s.pict_type == AV_PICTURE_TYPE_BI)) {

        v->x8_type = get_bits1(gb);

    } else

        v->x8_type = 0;

    av_dlog(v->s.avctx, "%c Frame: QP=[%i]%i (+%i/2) %i\n",

            (v->s.pict_type == AV_PICTURE_TYPE_P) ? 'P' : ((v->s.pict_type == AV_PICTURE_TYPE_I) ? 'I' : 'B'),

            pqindex, v->pq, v->halfpq, v->rangeredfrm);



    if (v->first_pic_header_flag)

        rotate_luts(v);



    switch (v->s.pict_type) {

    case AV_PICTURE_TYPE_P:

        if (v->pq < 5)       v->tt_index = 0;

        else if (v->pq < 13) v->tt_index = 1;

        else                 v->tt_index = 2;



        lowquant = (v->pq > 12) ? 0 : 1;

        v->mv_mode = ff_vc1_mv_pmode_table[lowquant][get_unary(gb, 1, 4)];

        if (v->mv_mode == MV_PMODE_INTENSITY_COMP) {

            v->mv_mode2 = ff_vc1_mv_pmode_table2[lowquant][get_unary(gb, 1, 3)];

            v->lumscale = get_bits(gb, 6);

            v->lumshift = get_bits(gb, 6);

            v->last_use_ic = 1;

            /* fill lookup tables for intensity compensation */

            INIT_LUT(v->lumscale, v->lumshift, v->last_luty[0], v->last_lutuv[0], 1);

            INIT_LUT(v->lumscale, v->lumshift, v->last_luty[1], v->last_lutuv[1], 1);

        }

        v->qs_last = v->s.quarter_sample;

        if (v->mv_mode == MV_PMODE_1MV_HPEL || v->mv_mode == MV_PMODE_1MV_HPEL_BILIN)

            v->s.quarter_sample = 0;

        else if (v->mv_mode == MV_PMODE_INTENSITY_COMP) {

            if (v->mv_mode2 == MV_PMODE_1MV_HPEL || v->mv_mode2 == MV_PMODE_1MV_HPEL_BILIN)

                v->s.quarter_sample = 0;

            else

                v->s.quarter_sample = 1;

        } else

            v->s.quarter_sample = 1;

        v->s.mspel = !(v->mv_mode == MV_PMODE_1MV_HPEL_BILIN || (v->mv_mode == MV_PMODE_INTENSITY_COMP && v->mv_mode2 == MV_PMODE_1MV_HPEL_BILIN));



        if ((v->mv_mode  == MV_PMODE_INTENSITY_COMP &&

             v->mv_mode2 == MV_PMODE_MIXED_MV)      ||

            v->mv_mode   == MV_PMODE_MIXED_MV) {

            status = bitplane_decoding(v->mv_type_mb_plane, &v->mv_type_is_raw, v);

            if (status < 0)

                return -1;

            av_log(v->s.avctx, AV_LOG_DEBUG, "MB MV Type plane encoding: "

                   "Imode: %i, Invert: %i\n", status>>1, status&1);

        } else {

            v->mv_type_is_raw = 0;

            memset(v->mv_type_mb_plane, 0, v->s.mb_stride * v->s.mb_height);

        }

        status = bitplane_decoding(v->s.mbskip_table, &v->skip_is_raw, v);

        if (status < 0)

            return -1;

        av_log(v->s.avctx, AV_LOG_DEBUG, "MB Skip plane encoding: "

               "Imode: %i, Invert: %i\n", status>>1, status&1);



        /* Hopefully this is correct for P frames */

        v->s.mv_table_index = get_bits(gb, 2); //but using ff_vc1_ tables

        v->cbpcy_vlc = &ff_vc1_cbpcy_p_vlc[get_bits(gb, 2)];



        if (v->dquant) {

            av_log(v->s.avctx, AV_LOG_DEBUG, "VOP DQuant info\n");

            vop_dquant_decoding(v);

        }



        v->ttfrm = 0; //FIXME Is that so ?

        if (v->vstransform) {

            v->ttmbf = get_bits1(gb);

            if (v->ttmbf) {

                v->ttfrm = ff_vc1_ttfrm_to_tt[get_bits(gb, 2)];

            }

        } else {

            v->ttmbf = 1;

            v->ttfrm = TT_8X8;

        }

        break;

    case AV_PICTURE_TYPE_B:

        if (v->pq < 5)       v->tt_index = 0;

        else if (v->pq < 13) v->tt_index = 1;

        else                 v->tt_index = 2;



        v->mv_mode          = get_bits1(gb) ? MV_PMODE_1MV : MV_PMODE_1MV_HPEL_BILIN;

        v->qs_last          = v->s.quarter_sample;

        v->s.quarter_sample = (v->mv_mode == MV_PMODE_1MV);

        v->s.mspel          = v->s.quarter_sample;



        status = bitplane_decoding(v->direct_mb_plane, &v->dmb_is_raw, v);

        if (status < 0)

            return -1;

        av_log(v->s.avctx, AV_LOG_DEBUG, "MB Direct Type plane encoding: "

               "Imode: %i, Invert: %i\n", status>>1, status&1);

        status = bitplane_decoding(v->s.mbskip_table, &v->skip_is_raw, v);

        if (status < 0)

            return -1;

        av_log(v->s.avctx, AV_LOG_DEBUG, "MB Skip plane encoding: "

               "Imode: %i, Invert: %i\n", status>>1, status&1);



        v->s.mv_table_index = get_bits(gb, 2);

        v->cbpcy_vlc        = &ff_vc1_cbpcy_p_vlc[get_bits(gb, 2)];



        if (v->dquant) {

            av_log(v->s.avctx, AV_LOG_DEBUG, "VOP DQuant info\n");

            vop_dquant_decoding(v);

        }



        v->ttfrm = 0;

        if (v->vstransform) {

            v->ttmbf = get_bits1(gb);

            if (v->ttmbf) {

                v->ttfrm = ff_vc1_ttfrm_to_tt[get_bits(gb, 2)];

            }

        } else {

            v->ttmbf = 1;

            v->ttfrm = TT_8X8;

        }

        break;

    }



    if (!v->x8_type) {

        /* AC Syntax */

        v->c_ac_table_index = decode012(gb);

        if (v->s.pict_type == AV_PICTURE_TYPE_I || v->s.pict_type == AV_PICTURE_TYPE_BI) {

            v->y_ac_table_index = decode012(gb);

        }

        /* DC Syntax */

        v->s.dc_table_index = get_bits1(gb);

    }



    if (v->s.pict_type == AV_PICTURE_TYPE_BI) {

        v->s.pict_type = AV_PICTURE_TYPE_B;

        v->bi_type     = 1;

    }

    return 0;

}