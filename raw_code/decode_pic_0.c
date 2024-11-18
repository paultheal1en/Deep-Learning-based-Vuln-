static int decode_pic(AVSContext *h) {

    MpegEncContext *s = &h->s;

    int skip_count;

    enum cavs_mb mb_type;



    if (!s->context_initialized) {

        s->avctx->idct_algo = FF_IDCT_CAVS;

        if (MPV_common_init(s) < 0)

            return -1;

        ff_init_scantable(s->dsp.idct_permutation,&h->scantable,ff_zigzag_direct);

    }

    skip_bits(&s->gb,16);//bbv_dwlay

    if(h->stc == PIC_PB_START_CODE) {

        h->pic_type = get_bits(&s->gb,2) + FF_I_TYPE;

        if(h->pic_type > FF_B_TYPE) {

            av_log(s->avctx, AV_LOG_ERROR, "illegal picture type\n");

            return -1;

        }

        /* make sure we have the reference frames we need */

        if(!h->DPB[0].data[0] ||

          (!h->DPB[1].data[0] && h->pic_type == FF_B_TYPE))

            return -1;

    } else {

        h->pic_type = FF_I_TYPE;

        if(get_bits1(&s->gb))

            skip_bits(&s->gb,24);//time_code

    }

    /* release last B frame */

    if(h->picture.data[0])

        s->avctx->release_buffer(s->avctx, (AVFrame *)&h->picture);



    s->avctx->get_buffer(s->avctx, (AVFrame *)&h->picture);

    ff_cavs_init_pic(h);

    h->picture.poc = get_bits(&s->gb,8)*2;



    /* get temporal distances and MV scaling factors */

    if(h->pic_type != FF_B_TYPE) {

        h->dist[0] = (h->picture.poc - h->DPB[0].poc  + 512) % 512;

    } else {

        h->dist[0] = (h->DPB[0].poc  - h->picture.poc + 512) % 512;

    }

    h->dist[1] = (h->picture.poc - h->DPB[1].poc  + 512) % 512;

    h->scale_den[0] = h->dist[0] ? 512/h->dist[0] : 0;

    h->scale_den[1] = h->dist[1] ? 512/h->dist[1] : 0;

    if(h->pic_type == FF_B_TYPE) {

        h->sym_factor = h->dist[0]*h->scale_den[1];

    } else {

        h->direct_den[0] = h->dist[0] ? 16384/h->dist[0] : 0;

        h->direct_den[1] = h->dist[1] ? 16384/h->dist[1] : 0;

    }



    if(s->low_delay)

        get_ue_golomb(&s->gb); //bbv_check_times

    h->progressive             = get_bits1(&s->gb);

    h->pic_structure = 1;

    if(!h->progressive)

        h->pic_structure = get_bits1(&s->gb);

    if(!h->pic_structure && h->stc == PIC_PB_START_CODE)

        skip_bits1(&s->gb);     //advanced_pred_mode_disable

    skip_bits1(&s->gb);        //top_field_first

    skip_bits1(&s->gb);        //repeat_first_field

    h->qp_fixed                = get_bits1(&s->gb);

    h->qp                      = get_bits(&s->gb,6);

    if(h->pic_type == FF_I_TYPE) {

        if(!h->progressive && !h->pic_structure)

            skip_bits1(&s->gb);//what is this?

        skip_bits(&s->gb,4);   //reserved bits

    } else {

        if(!(h->pic_type == FF_B_TYPE && h->pic_structure == 1))

            h->ref_flag        = get_bits1(&s->gb);

        skip_bits(&s->gb,4);   //reserved bits

        h->skip_mode_flag      = get_bits1(&s->gb);

    }

    h->loop_filter_disable     = get_bits1(&s->gb);

    if(!h->loop_filter_disable && get_bits1(&s->gb)) {

        h->alpha_offset        = get_se_golomb(&s->gb);

        h->beta_offset         = get_se_golomb(&s->gb);

    } else {

        h->alpha_offset = h->beta_offset  = 0;

    }

    if(h->pic_type == FF_I_TYPE) {

        do {

            check_for_slice(h);

            decode_mb_i(h, 0);

        } while(ff_cavs_next_mb(h));

    } else if(h->pic_type == FF_P_TYPE) {

        do {

            check_for_slice(h);

            if(h->skip_mode_flag) {

                skip_count = get_ue_golomb(&s->gb);

                while(skip_count--) {

                    decode_mb_p(h,P_SKIP);

                    if(!ff_cavs_next_mb(h))

                        goto done;

                }

                check_for_slice(h);

                mb_type = get_ue_golomb(&s->gb) + P_16X16;

            } else

                mb_type = get_ue_golomb(&s->gb) + P_SKIP;

            if(mb_type > P_8X8) {

                decode_mb_i(h, mb_type - P_8X8 - 1);

            } else

                decode_mb_p(h,mb_type);

        } while(ff_cavs_next_mb(h));

    } else { /* FF_B_TYPE */

        do {

            check_for_slice(h);

            if(h->skip_mode_flag) {

                skip_count = get_ue_golomb(&s->gb);

                while(skip_count--) {

                    decode_mb_b(h,B_SKIP);

                    if(!ff_cavs_next_mb(h))

                        goto done;

                }

                check_for_slice(h);

                mb_type = get_ue_golomb(&s->gb) + B_DIRECT;

            } else

                mb_type = get_ue_golomb(&s->gb) + B_SKIP;

            if(mb_type > B_8X8) {

                decode_mb_i(h, mb_type - B_8X8 - 1);

            } else

                decode_mb_b(h,mb_type);

        } while(ff_cavs_next_mb(h));

    }

 done:

    if(h->pic_type != FF_B_TYPE) {

        if(h->DPB[1].data[0])

            s->avctx->release_buffer(s->avctx, (AVFrame *)&h->DPB[1]);

        h->DPB[1] = h->DPB[0];

        h->DPB[0] = h->picture;

        memset(&h->picture,0,sizeof(Picture));

    }

    return 0;

}
