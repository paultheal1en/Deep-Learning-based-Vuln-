static int h264_mp4toannexb_filter(AVBitStreamFilterContext *bsfc,

                                   AVCodecContext *avctx, const char *args,

                                   uint8_t  **poutbuf, int *poutbuf_size,

                                   const uint8_t *buf, int      buf_size,

                                   int keyframe) {

    H264BSFContext *ctx = bsfc->priv_data;

    uint8_t unit_type;

    uint32_t nal_size, cumul_size = 0;



    /* nothing to filter */

    if (!avctx->extradata || avctx->extradata_size < 6) {

        *poutbuf = (uint8_t*) buf;

        *poutbuf_size = buf_size;

        return 0;

    }



    /* retrieve sps and pps NAL units from extradata */

    if (!ctx->sps_pps_data) {

        uint16_t unit_size;

        uint32_t total_size = 0;

        uint8_t *out = NULL, unit_nb, sps_done = 0;

        const uint8_t *extradata = avctx->extradata+4;

        static const uint8_t nalu_header[4] = {0, 0, 0, 1};



        /* retrieve length coded size */

        ctx->length_size = (*extradata++ & 0x3) + 1;

        if (ctx->length_size == 3)

            return AVERROR(EINVAL);



        /* retrieve sps and pps unit(s) */

        unit_nb = *extradata++ & 0x1f; /* number of sps unit(s) */

        if (!unit_nb) {

            unit_nb = *extradata++; /* number of pps unit(s) */

            sps_done++;

        }

        while (unit_nb--) {

            unit_size = AV_RB16(extradata);

            total_size += unit_size+4;

            if (extradata+2+unit_size > avctx->extradata+avctx->extradata_size) {

                av_free(out);

                return AVERROR(EINVAL);

            }

            out = av_realloc(out, total_size);

            if (!out)

                return AVERROR(ENOMEM);

            memcpy(out+total_size-unit_size-4, nalu_header, 4);

            memcpy(out+total_size-unit_size,   extradata+2, unit_size);

            extradata += 2+unit_size;



            if (!unit_nb && !sps_done++)

                unit_nb = *extradata++; /* number of pps unit(s) */

        }



        ctx->sps_pps_data = out;

        ctx->size = total_size;

        ctx->first_idr = 1;

    }



    *poutbuf_size = 0;

    *poutbuf = NULL;

    do {

        if (ctx->length_size == 1)

            nal_size = buf[0];

        else if (ctx->length_size == 2)

            nal_size = AV_RB16(buf);

        else

            nal_size = AV_RB32(buf);



        buf += ctx->length_size;

        unit_type = *buf & 0x1f;



        /* prepend only to the first type 5 NAL unit of an IDR picture */

        if (ctx->first_idr && unit_type == 5) {

            alloc_and_copy(poutbuf, poutbuf_size,

                           ctx->sps_pps_data, ctx->size,

                           buf, nal_size);

            ctx->first_idr = 0;

        }

        else {

            alloc_and_copy(poutbuf, poutbuf_size,

                           NULL, 0,

                           buf, nal_size);

            if (!ctx->first_idr && unit_type == 1)

                ctx->first_idr = 1;

        }



        buf += nal_size;

        cumul_size += nal_size + ctx->length_size;

    } while (cumul_size < buf_size);



    return 1;

}
