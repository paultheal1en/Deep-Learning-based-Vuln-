static av_cold int svq3_decode_init(AVCodecContext *avctx)

{

    SVQ3Context *svq3 = avctx->priv_data;

    H264Context *h = &svq3->h;

    MpegEncContext *s = &h->s;

    int m;

    unsigned char *extradata;

    unsigned char *extradata_end;

    unsigned int size;

    int marker_found = 0;



    if (ff_h264_decode_init(avctx) < 0)

        return -1;



    s->flags  = avctx->flags;

    s->flags2 = avctx->flags2;

    s->unrestricted_mv = 1;

    h->is_complex=1;

    h->sps.chroma_format_idc = 1;

    avctx->pix_fmt = avctx->codec->pix_fmts[0];



    if (!s->context_initialized) {

        h->chroma_qp[0] = h->chroma_qp[1] = 4;



        svq3->halfpel_flag  = 1;

        svq3->thirdpel_flag = 1;

        svq3->unknown_flag  = 0;





        /* prowl for the "SEQH" marker in the extradata */

        extradata = (unsigned char *)avctx->extradata;

        extradata_end = avctx->extradata + avctx->extradata_size;

        if (extradata) {

            for (m = 0; m + 8 < avctx->extradata_size; m++) {

                if (!memcmp(extradata, "SEQH", 4)) {

                    marker_found = 1;

                    break;

                }

                extradata++;

            }

        }



        /* if a match was found, parse the extra data */

        if (marker_found) {



            GetBitContext gb;

            int frame_size_code;



            size = AV_RB32(&extradata[4]);

            if (size > extradata_end - extradata - 8)

                return AVERROR_INVALIDDATA;

            init_get_bits(&gb, extradata + 8, size*8);



            /* 'frame size code' and optional 'width, height' */

            frame_size_code = get_bits(&gb, 3);

            switch (frame_size_code) {

                case 0: avctx->width = 160; avctx->height = 120; break;

                case 1: avctx->width = 128; avctx->height =  96; break;

                case 2: avctx->width = 176; avctx->height = 144; break;

                case 3: avctx->width = 352; avctx->height = 288; break;

                case 4: avctx->width = 704; avctx->height = 576; break;

                case 5: avctx->width = 240; avctx->height = 180; break;

                case 6: avctx->width = 320; avctx->height = 240; break;

                case 7:

                    avctx->width  = get_bits(&gb, 12);

                    avctx->height = get_bits(&gb, 12);

                    break;

            }



            svq3->halfpel_flag  = get_bits1(&gb);

            svq3->thirdpel_flag = get_bits1(&gb);



            /* unknown fields */

            skip_bits1(&gb);

            skip_bits1(&gb);

            skip_bits1(&gb);

            skip_bits1(&gb);



            s->low_delay = get_bits1(&gb);



            /* unknown field */

            skip_bits1(&gb);



            while (get_bits1(&gb)) {

                skip_bits(&gb, 8);

            }



            svq3->unknown_flag = get_bits1(&gb);

            avctx->has_b_frames = !s->low_delay;

            if (svq3->unknown_flag) {

#if CONFIG_ZLIB

                unsigned watermark_width  = svq3_get_ue_golomb(&gb);

                unsigned watermark_height = svq3_get_ue_golomb(&gb);

                int u1 = svq3_get_ue_golomb(&gb);

                int u2 = get_bits(&gb, 8);

                int u3 = get_bits(&gb, 2);

                int u4 = svq3_get_ue_golomb(&gb);

                unsigned long buf_len = watermark_width*watermark_height*4;

                int offset = (get_bits_count(&gb)+7)>>3;

                uint8_t *buf;



                if ((uint64_t)watermark_width*4 > UINT_MAX/watermark_height)

                    return -1;



                buf = av_malloc(buf_len);

                av_log(avctx, AV_LOG_DEBUG, "watermark size: %dx%d\n", watermark_width, watermark_height);

                av_log(avctx, AV_LOG_DEBUG, "u1: %x u2: %x u3: %x compressed data size: %d offset: %d\n", u1, u2, u3, u4, offset);

                if (uncompress(buf, &buf_len, extradata + 8 + offset, size - offset) != Z_OK) {

                    av_log(avctx, AV_LOG_ERROR, "could not uncompress watermark logo\n");

                    av_free(buf);

                    return -1;

                }

                svq3->watermark_key = ff_svq1_packet_checksum(buf, buf_len, 0);

                svq3->watermark_key = svq3->watermark_key << 16 | svq3->watermark_key;

                av_log(avctx, AV_LOG_DEBUG, "watermark key %#x\n", svq3->watermark_key);

                av_free(buf);

#else

                av_log(avctx, AV_LOG_ERROR, "this svq3 file contains watermark which need zlib support compiled in\n");

                return -1;

#endif

            }

        }



        s->width  = avctx->width;

        s->height = avctx->height;



        if (ff_MPV_common_init(s) < 0)

            return -1;



        h->b_stride = 4*s->mb_width;



        if (ff_h264_alloc_tables(h) < 0) {

            av_log(avctx, AV_LOG_ERROR, "svq3 memory allocation failed\n");

            return AVERROR(ENOMEM);

        }

    }



    return 0;

}
