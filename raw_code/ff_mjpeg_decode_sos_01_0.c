int ff_mjpeg_decode_sos(MJpegDecodeContext *s,

                        const uint8_t *mb_bitmask, const AVFrame *reference)

{

    int len, nb_components, i, h, v, predictor, point_transform;

    int index, id;

    const int block_size= s->lossless ? 1 : 8;

    int ilv, prev_shift;



    /* XXX: verify len field validity */

    len = get_bits(&s->gb, 16);

    nb_components = get_bits(&s->gb, 8);

    if (nb_components == 0 || nb_components > MAX_COMPONENTS){

        av_log(s->avctx, AV_LOG_ERROR, "decode_sos: nb_components (%d) unsupported\n", nb_components);

        return -1;

    }

    if (len != 6+2*nb_components)

    {

        av_log(s->avctx, AV_LOG_ERROR, "decode_sos: invalid len (%d)\n", len);

        return -1;

    }

    for(i=0;i<nb_components;i++) {

        id = get_bits(&s->gb, 8) - 1;

        av_log(s->avctx, AV_LOG_DEBUG, "component: %d\n", id);

        /* find component index */

        for(index=0;index<s->nb_components;index++)

            if (id == s->component_id[index])

                break;

        if (index == s->nb_components)

        {

            av_log(s->avctx, AV_LOG_ERROR, "decode_sos: index(%d) out of components\n", index);

            return -1;

        }

        /* Metasoft MJPEG codec has Cb and Cr swapped */

        if (s->avctx->codec_tag == MKTAG('M', 'T', 'S', 'J')

            && nb_components == 3 && s->nb_components == 3 && i)

            index = 3 - i;



        if(nb_components == 3 && s->nb_components == 3 && s->avctx->pix_fmt == PIX_FMT_GBR24P)

            index = (i+2)%3;



        s->comp_index[i] = index;



        s->nb_blocks[i] = s->h_count[index] * s->v_count[index];

        s->h_scount[i] = s->h_count[index];

        s->v_scount[i] = s->v_count[index];



        s->dc_index[i] = get_bits(&s->gb, 4);

        s->ac_index[i] = get_bits(&s->gb, 4);



        if (s->dc_index[i] <  0 || s->ac_index[i] < 0 ||

            s->dc_index[i] >= 4 || s->ac_index[i] >= 4)

            goto out_of_range;

        if (!s->vlcs[0][s->dc_index[i]].table || !s->vlcs[1][s->ac_index[i]].table)

            goto out_of_range;

    }



    predictor= get_bits(&s->gb, 8); /* JPEG Ss / lossless JPEG predictor /JPEG-LS NEAR */

    ilv= get_bits(&s->gb, 8);    /* JPEG Se / JPEG-LS ILV */

    if(s->avctx->codec_tag != AV_RL32("CJPG")){

        prev_shift = get_bits(&s->gb, 4); /* Ah */

        point_transform= get_bits(&s->gb, 4); /* Al */

    }else

        prev_shift= point_transform= 0;



    for(i=0;i<nb_components;i++)

        s->last_dc[i] = 1024;



    if (nb_components > 1) {

        /* interleaved stream */

        s->mb_width  = (s->width  + s->h_max * block_size - 1) / (s->h_max * block_size);

        s->mb_height = (s->height + s->v_max * block_size - 1) / (s->v_max * block_size);

    } else if(!s->ls) { /* skip this for JPEG-LS */

        h = s->h_max / s->h_scount[0];

        v = s->v_max / s->v_scount[0];

        s->mb_width  = (s->width  + h * block_size - 1) / (h * block_size);

        s->mb_height = (s->height + v * block_size - 1) / (v * block_size);

        s->nb_blocks[0] = 1;

        s->h_scount[0] = 1;

        s->v_scount[0] = 1;

    }



    if(s->avctx->debug & FF_DEBUG_PICT_INFO)

        av_log(s->avctx, AV_LOG_DEBUG, "%s %s p:%d >>:%d ilv:%d bits:%d skip:%d %s comp:%d\n", s->lossless ? "lossless" : "sequential DCT", s->rgb ? "RGB" : "",

               predictor, point_transform, ilv, s->bits, s->mjpb_skiptosod,

               s->pegasus_rct ? "PRCT" : (s->rct ? "RCT" : ""), nb_components);





    /* mjpeg-b can have padding bytes between sos and image data, skip them */

    for (i = s->mjpb_skiptosod; i > 0; i--)

        skip_bits(&s->gb, 8);



    if(s->lossless){

        av_assert0(s->picture_ptr == &s->picture);

        if(CONFIG_JPEGLS_DECODER && s->ls){

//            for(){

//            reset_ls_coding_parameters(s, 0);



            if(ff_jpegls_decode_picture(s, predictor, point_transform, ilv) < 0)

                return -1;

        }else{

            if(s->rgb){

                if(ljpeg_decode_rgb_scan(s, nb_components, predictor, point_transform) < 0)

                    return -1;

            }else{

                if(ljpeg_decode_yuv_scan(s, predictor, point_transform) < 0)

                    return -1;

            }

        }

    }else{

        if(s->progressive && predictor) {

            av_assert0(s->picture_ptr == &s->picture);

            if(mjpeg_decode_scan_progressive_ac(s, predictor, ilv, prev_shift, point_transform) < 0)

                return -1;

        } else {

            if(mjpeg_decode_scan(s, nb_components, prev_shift, point_transform,

                                 mb_bitmask, reference) < 0)

                return -1;

        }

    }

    if (s->yuv421) {

        uint8_t *line = s->picture_ptr->data[2];

        for (i = 0; i < s->height / 2; i++) {

            for (index = s->width - 1; index; index--)

                line[index] = (line[index / 2] + line[(index + 1) / 2]) >> 1;

            line += s->linesize[2];

        }

    } else if (s->yuv442) {

        uint8_t *dst = &((uint8_t *)s->picture_ptr->data[2])[(s->height - 1) * s->linesize[2]];

        for (i = s->height - 1; i; i--) {

            uint8_t *src1 = &((uint8_t *)s->picture_ptr->data[2])[i / 2 * s->linesize[2]];

            uint8_t *src2 = &((uint8_t *)s->picture_ptr->data[2])[(i + 1) / 2 * s->linesize[2]];

            if (src1 == src2) {

                memcpy(dst, src1, s->width);

            } else {

                for (index = 0; index < s->width; index++)

                    dst[index] = (src1[index] + src2[index]) >> 1;

            }

            dst -= s->linesize[2];

        }

    }

    emms_c();

    return 0;

 out_of_range:

    av_log(s->avctx, AV_LOG_ERROR, "decode_sos: ac/dc index out of range\n");

    return -1;

}
