static int decode_band_hdr(IVI45DecContext *ctx, IVIBandDesc *band,

                           AVCodecContext *avctx)

{

    int plane, band_num, indx, transform_id, scan_indx;

    int i;



    plane    = get_bits(&ctx->gb, 2);

    band_num = get_bits(&ctx->gb, 4);

    if (band->plane != plane || band->band_num != band_num) {

        av_log(avctx, AV_LOG_ERROR, "Invalid band header sequence!\n");

        return AVERROR_INVALIDDATA;

    }



    band->is_empty = get_bits1(&ctx->gb);

    if (!band->is_empty) {

        int old_blk_size = band->blk_size;

        /* skip header size

         * If header size is not given, header size is 4 bytes. */

        if (get_bits1(&ctx->gb))

            skip_bits(&ctx->gb, 16);



        band->is_halfpel = get_bits(&ctx->gb, 2);

        if (band->is_halfpel >= 2) {

            av_log(avctx, AV_LOG_ERROR, "Invalid/unsupported mv resolution: %d!\n",

                   band->is_halfpel);

            return AVERROR_INVALIDDATA;

        }

#if IVI4_STREAM_ANALYSER

        if (!band->is_halfpel)

            ctx->uses_fullpel = 1;

#endif



        band->checksum_present = get_bits1(&ctx->gb);

        if (band->checksum_present)

            band->checksum = get_bits(&ctx->gb, 16);



        indx = get_bits(&ctx->gb, 2);

        if (indx == 3) {

            av_log(avctx, AV_LOG_ERROR, "Invalid block size!\n");

            return AVERROR_INVALIDDATA;

        }

        band->mb_size  = 16 >> indx;

        band->blk_size = 8 >> (indx >> 1);



        band->inherit_mv     = get_bits1(&ctx->gb);

        band->inherit_qdelta = get_bits1(&ctx->gb);



        band->glob_quant = get_bits(&ctx->gb, 5);



        if (!get_bits1(&ctx->gb) || ctx->frame_type == IVI4_FRAMETYPE_INTRA) {

            transform_id = get_bits(&ctx->gb, 5);

            if (transform_id >= FF_ARRAY_ELEMS(transforms) ||

                !transforms[transform_id].inv_trans) {

                avpriv_request_sample(avctx, "Transform %d", transform_id);

                return AVERROR_PATCHWELCOME;

            }

            if ((transform_id >= 7 && transform_id <= 9) ||

                 transform_id == 17) {

                avpriv_request_sample(avctx, "DCT transform");

                return AVERROR_PATCHWELCOME;

            }



#if IVI4_STREAM_ANALYSER

            if ((transform_id >= 0 && transform_id <= 2) || transform_id == 10)

                ctx->uses_haar = 1;

#endif



            band->inv_transform = transforms[transform_id].inv_trans;

            band->dc_transform  = transforms[transform_id].dc_trans;

            band->is_2d_trans   = transforms[transform_id].is_2d_trans;

            if (transform_id < 10)

                band->transform_size = 8;

            else

                band->transform_size = 4;



            if (band->blk_size != band->transform_size)

                return AVERROR_INVALIDDATA;



            scan_indx = get_bits(&ctx->gb, 4);

            if (scan_indx == 15) {

                av_log(avctx, AV_LOG_ERROR, "Custom scan pattern encountered!\n");

                return AVERROR_INVALIDDATA;

            }

            if (scan_indx > 4 && scan_indx < 10) {

                if (band->blk_size != 4)

                    return AVERROR_INVALIDDATA;

            } else if (band->blk_size != 8)

                return AVERROR_INVALIDDATA;



            band->scan = scan_index_to_tab[scan_indx];



            band->quant_mat = get_bits(&ctx->gb, 5);

            if (band->quant_mat >= FF_ARRAY_ELEMS(quant_index_to_tab)) {



                if (band->quant_mat == 31)

                    av_log(avctx, AV_LOG_ERROR,

                           "Custom quant matrix encountered!\n");

                else

                    avpriv_request_sample(avctx, "Quantization matrix %d",

                                          band->quant_mat);

                band->quant_mat = -1;

                return AVERROR_INVALIDDATA;

            }

        } else {

            if (old_blk_size != band->blk_size) {

                av_log(avctx, AV_LOG_ERROR,

                       "The band block size does not match the configuration "

                       "inherited\n");

                return AVERROR_INVALIDDATA;

            }

            if (band->quant_mat < 0) {

                av_log(avctx, AV_LOG_ERROR, "Invalid quant_mat inherited\n");

                return AVERROR_INVALIDDATA;

            }

        }



        /* decode block huffman codebook */

        if (!get_bits1(&ctx->gb))

            band->blk_vlc.tab = ctx->blk_vlc.tab;

        else

            if (ff_ivi_dec_huff_desc(&ctx->gb, 1, IVI_BLK_HUFF,

                                     &band->blk_vlc, avctx))

                return AVERROR_INVALIDDATA;



        /* select appropriate rvmap table for this band */

        band->rvmap_sel = get_bits1(&ctx->gb) ? get_bits(&ctx->gb, 3) : 8;



        /* decode rvmap probability corrections if any */

        band->num_corr = 0; /* there is no corrections */

        if (get_bits1(&ctx->gb)) {

            band->num_corr = get_bits(&ctx->gb, 8); /* get number of correction pairs */

            if (band->num_corr > 61) {

                av_log(avctx, AV_LOG_ERROR, "Too many corrections: %d\n",

                       band->num_corr);

                return AVERROR_INVALIDDATA;

            }



            /* read correction pairs */

            for (i = 0; i < band->num_corr * 2; i++)

                band->corr[i] = get_bits(&ctx->gb, 8);

        }

    }



    if (band->blk_size == 8) {

        band->intra_base = &ivi4_quant_8x8_intra[quant_index_to_tab[band->quant_mat]][0];

        band->inter_base = &ivi4_quant_8x8_inter[quant_index_to_tab[band->quant_mat]][0];

    } else {

        band->intra_base = &ivi4_quant_4x4_intra[quant_index_to_tab[band->quant_mat]][0];

        band->inter_base = &ivi4_quant_4x4_inter[quant_index_to_tab[band->quant_mat]][0];

    }



    /* Indeo 4 doesn't use scale tables */

    band->intra_scale = NULL;

    band->inter_scale = NULL;



    align_get_bits(&ctx->gb);



    return 0;

}
