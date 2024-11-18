static int dvbsub_decode(AVCodecContext *avctx,

                         void *data, int *data_size,

                         AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    DVBSubContext *ctx = avctx->priv_data;

    AVSubtitle *sub = data;

    const uint8_t *p, *p_end;

    int segment_type;

    int page_id;

    int segment_length;



#ifdef DEBUG_PACKET_CONTENTS

    int i;



    av_log(avctx, AV_LOG_INFO, "DVB sub packet:\n");



    for (i=0; i < buf_size; i++) {

        av_log(avctx, AV_LOG_INFO, "%02x ", buf[i]);

        if (i % 16 == 15)

            av_log(avctx, AV_LOG_INFO, "\n");

    }



    if (i % 16)

        av_log(avctx, AV_LOG_INFO, "\n");



#endif



    if (buf_size <= 2 || *buf != 0x0f)

        return -1;



    p = buf;

    p_end = buf + buf_size;



    while (p < p_end && *p == 0x0f) {

        p += 1;

        segment_type = *p++;

        page_id = AV_RB16(p);

        p += 2;

        segment_length = AV_RB16(p);

        p += 2;



        if (page_id == ctx->composition_id || page_id == ctx->ancillary_id ||

            ctx->composition_id == -1 || ctx->ancillary_id == -1) {

            switch (segment_type) {

            case DVBSUB_PAGE_SEGMENT:

                dvbsub_parse_page_segment(avctx, p, segment_length);

                break;

            case DVBSUB_REGION_SEGMENT:

                dvbsub_parse_region_segment(avctx, p, segment_length);

                break;

            case DVBSUB_CLUT_SEGMENT:

                dvbsub_parse_clut_segment(avctx, p, segment_length);

                break;

            case DVBSUB_OBJECT_SEGMENT:

                dvbsub_parse_object_segment(avctx, p, segment_length);

                break;

            case DVBSUB_DISPLAYDEFINITION_SEGMENT:

                dvbsub_parse_display_definition_segment(avctx, p, segment_length);

            case DVBSUB_DISPLAY_SEGMENT:

                *data_size = dvbsub_display_end_segment(avctx, p, segment_length, sub);

                break;

            default:

                av_dlog(avctx, "Subtitling segment type 0x%x, page id %d, length %d\n",

                        segment_type, page_id, segment_length);

                break;

            }

        }



        p += segment_length;

    }



    return p - buf;

}
