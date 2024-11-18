static int extract_extradata_h2645(AVBSFContext *ctx, AVPacket *pkt,

                                   uint8_t **data, int *size)

{

    static const int extradata_nal_types_hevc[] = {

        HEVC_NAL_VPS, HEVC_NAL_SPS, HEVC_NAL_PPS,

    };

    static const int extradata_nal_types_h264[] = {

        H264_NAL_SPS, H264_NAL_PPS,

    };



    ExtractExtradataContext *s = ctx->priv_data;



    H2645Packet h2645_pkt = { 0 };

    int extradata_size = 0;

    const int *extradata_nal_types;

    int nb_extradata_nal_types;

    int i, has_sps = 0, has_vps = 0, ret = 0;



    if (ctx->par_in->codec_id == AV_CODEC_ID_HEVC) {

        extradata_nal_types    = extradata_nal_types_hevc;

        nb_extradata_nal_types = FF_ARRAY_ELEMS(extradata_nal_types_hevc);

    } else {

        extradata_nal_types    = extradata_nal_types_h264;

        nb_extradata_nal_types = FF_ARRAY_ELEMS(extradata_nal_types_h264);

    }



    ret = ff_h2645_packet_split(&h2645_pkt, pkt->data, pkt->size,

                                ctx, 0, 0, ctx->par_in->codec_id, 1);

    if (ret < 0)

        return ret;



    for (i = 0; i < h2645_pkt.nb_nals; i++) {

        H2645NAL *nal = &h2645_pkt.nals[i];

        if (val_in_array(extradata_nal_types, nb_extradata_nal_types, nal->type)) {

            extradata_size += nal->raw_size + 3;

            if (ctx->par_in->codec_id == AV_CODEC_ID_HEVC) {

                if (nal->type == HEVC_NAL_SPS) has_sps = 1;

                if (nal->type == HEVC_NAL_VPS) has_vps = 1;

            } else {

                if (nal->type == H264_NAL_SPS) has_sps = 1;

            }

        }

    }



    if (extradata_size &&

        ((ctx->par_in->codec_id == AV_CODEC_ID_HEVC && has_sps && has_vps) ||

         (ctx->par_in->codec_id == AV_CODEC_ID_H264 && has_sps))) {

        AVBufferRef *filtered_buf;

        uint8_t *extradata, *filtered_data;



        if (s->remove) {

            filtered_buf = av_buffer_alloc(pkt->size + AV_INPUT_BUFFER_PADDING_SIZE);

            if (!filtered_buf) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            filtered_data = filtered_buf->data;

        }



        extradata = av_malloc(extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);

        if (!extradata) {

            av_buffer_unref(&filtered_buf);

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        *data = extradata;

        *size = extradata_size;



        for (i = 0; i < h2645_pkt.nb_nals; i++) {

            H2645NAL *nal = &h2645_pkt.nals[i];

            if (val_in_array(extradata_nal_types, nb_extradata_nal_types,

                             nal->type)) {

                AV_WB24(extradata, 1); // startcode

                memcpy(extradata + 3, nal->raw_data, nal->raw_size);

                extradata += 3 + nal->raw_size;

            } else if (s->remove) {

                AV_WB24(filtered_data, 1); // startcode

                memcpy(filtered_data + 3, nal->raw_data, nal->raw_size);

                filtered_data += 3 + nal->raw_size;

            }

        }



        if (s->remove) {

            av_buffer_unref(&pkt->buf);

            pkt->buf  = filtered_buf;

            pkt->data = filtered_buf->data;

            pkt->size = filtered_data - filtered_buf->data;

        }

    }



fail:

    ff_h2645_packet_uninit(&h2645_pkt);

    return ret;

}
