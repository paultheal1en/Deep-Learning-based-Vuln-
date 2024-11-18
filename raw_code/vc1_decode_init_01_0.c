static int vc1_decode_init(AVCodecContext *avctx)

{

    VC1Context *v = avctx->priv_data;

    MpegEncContext *s = &v->s;

    GetBitContext gb;



    if (!avctx->extradata_size || !avctx->extradata) return -1;

    if (!(avctx->flags & CODEC_FLAG_GRAY))

        avctx->pix_fmt = PIX_FMT_YUV420P;

    else

        avctx->pix_fmt = PIX_FMT_GRAY8;

    v->s.avctx = avctx;

    avctx->flags |= CODEC_FLAG_EMU_EDGE;

    v->s.flags |= CODEC_FLAG_EMU_EDGE;



    if(avctx->idct_algo==FF_IDCT_AUTO){

        avctx->idct_algo=FF_IDCT_WMV2;

    }



    if(ff_h263_decode_init(avctx) < 0)

        return -1;

    if (vc1_init_common(v) < 0) return -1;



    avctx->coded_width = avctx->width;

    avctx->coded_height = avctx->height;

    if (avctx->codec_id == CODEC_ID_WMV3)

    {

        int count = 0;



        // looks like WMV3 has a sequence header stored in the extradata

        // advanced sequence header may be before the first frame

        // the last byte of the extradata is a version number, 1 for the

        // samples we can decode



        init_get_bits(&gb, avctx->extradata, avctx->extradata_size*8);



        if (decode_sequence_header(avctx, &gb) < 0)

          return -1;



        count = avctx->extradata_size*8 - get_bits_count(&gb);

        if (count>0)

        {

            av_log(avctx, AV_LOG_INFO, "Extra data: %i bits left, value: %X\n",

                   count, get_bits(&gb, count));

        }

        else if (count < 0)

        {

            av_log(avctx, AV_LOG_INFO, "Read %i bits in overflow\n", -count);

        }

    } else { // VC1/WVC1

        const uint8_t *start = avctx->extradata;

        uint8_t *end = avctx->extradata + avctx->extradata_size;

        const uint8_t *next;

        int size, buf2_size;

        uint8_t *buf2 = NULL;

        int seq_inited = 0, ep_inited = 0;



        if(avctx->extradata_size < 16) {

            av_log(avctx, AV_LOG_ERROR, "Extradata size too small: %i\n", avctx->extradata_size);

            return -1;

        }



        buf2 = av_mallocz(avctx->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

        if(start[0]) start++; // in WVC1 extradata first byte is its size

        next = start;

        for(; next < end; start = next){

            next = find_next_marker(start + 4, end);

            size = next - start - 4;

            if(size <= 0) continue;

            buf2_size = vc1_unescape_buffer(start + 4, size, buf2);

            init_get_bits(&gb, buf2, buf2_size * 8);

            switch(AV_RB32(start)){

            case VC1_CODE_SEQHDR:

                if(decode_sequence_header(avctx, &gb) < 0){

                    av_free(buf2);

                    return -1;

                }

                seq_inited = 1;

                break;

            case VC1_CODE_ENTRYPOINT:

                if(decode_entry_point(avctx, &gb) < 0){

                    av_free(buf2);

                    return -1;

                }

                ep_inited = 1;

                break;

            }

        }

        av_free(buf2);

        if(!seq_inited || !ep_inited){

            av_log(avctx, AV_LOG_ERROR, "Incomplete extradata\n");

            return -1;

        }

    }

    avctx->has_b_frames= !!(avctx->max_b_frames);

    s->low_delay = !avctx->has_b_frames;



    s->mb_width = (avctx->coded_width+15)>>4;

    s->mb_height = (avctx->coded_height+15)>>4;



    /* Allocate mb bitplanes */

    v->mv_type_mb_plane = av_malloc(s->mb_stride * s->mb_height);

    v->direct_mb_plane = av_malloc(s->mb_stride * s->mb_height);

    v->acpred_plane = av_malloc(s->mb_stride * s->mb_height);

    v->over_flags_plane = av_malloc(s->mb_stride * s->mb_height);



    /* allocate block type info in that way so it could be used with s->block_index[] */

    v->mb_type_base = av_malloc(s->b8_stride * (s->mb_height * 2 + 1) + s->mb_stride * (s->mb_height + 1) * 2);

    v->mb_type[0] = v->mb_type_base + s->b8_stride + 1;

    v->mb_type[1] = v->mb_type_base + s->b8_stride * (s->mb_height * 2 + 1) + s->mb_stride + 1;

    v->mb_type[2] = v->mb_type[1] + s->mb_stride * (s->mb_height + 1);



    /* Init coded blocks info */

    if (v->profile == PROFILE_ADVANCED)

    {

//        if (alloc_bitplane(&v->over_flags_plane, s->mb_width, s->mb_height) < 0)

//            return -1;

//        if (alloc_bitplane(&v->ac_pred_plane, s->mb_width, s->mb_height) < 0)

//            return -1;

    }



    ff_intrax8_common_init(&v->x8,s);

    return 0;

}
