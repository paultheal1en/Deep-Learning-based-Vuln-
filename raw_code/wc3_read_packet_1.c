static int wc3_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    Wc3DemuxContext *wc3 = s->priv_data;

    ByteIOContext *pb = s->pb;

    unsigned int fourcc_tag;

    unsigned int size;

    int packet_read = 0;

    int ret = 0;

    unsigned char text[1024];

    unsigned int palette_number;

    int i;

    unsigned char r, g, b;

    int base_palette_index;



    while (!packet_read) {



        fourcc_tag = get_le32(pb);

        /* chunk sizes are 16-bit aligned */

        size = (get_be32(pb) + 1) & (~1);

        if (url_feof(pb))

            return AVERROR(EIO);



        switch (fourcc_tag) {



        case BRCH_TAG:

            /* no-op */

            break;



        case SHOT_TAG:

            /* load up new palette */

            palette_number = get_le32(pb);

            if (palette_number >= wc3->palette_count)

                return AVERROR_INVALIDDATA;

            base_palette_index = palette_number * PALETTE_COUNT * 3;

            for (i = 0; i < PALETTE_COUNT; i++) {

                r = wc3->palettes[base_palette_index + i * 3 + 0];

                g = wc3->palettes[base_palette_index + i * 3 + 1];

                b = wc3->palettes[base_palette_index + i * 3 + 2];

                wc3->palette_control.palette[i] = (r << 16) | (g << 8) | (b);

            }

            wc3->palette_control.palette_changed = 1;

            break;



        case VGA__TAG:

            /* send out video chunk */

            ret= av_get_packet(pb, pkt, size);

            pkt->stream_index = wc3->video_stream_index;

            pkt->pts = wc3->pts;

            packet_read = 1;

            break;



        case TEXT_TAG:

            /* subtitle chunk */

#if 0

            url_fseek(pb, size, SEEK_CUR);

#else

            if ((unsigned)size > sizeof(text) || (ret = get_buffer(pb, text, size)) != size)

                ret = AVERROR(EIO);

            else {

                int i = 0;

                av_log (s, AV_LOG_DEBUG, "Subtitle time!\n");

                av_log (s, AV_LOG_DEBUG, "  inglish: %s\n", &text[i + 1]);

                i += text[i] + 1;

                av_log (s, AV_LOG_DEBUG, "  doytsch: %s\n", &text[i + 1]);

                i += text[i] + 1;

                av_log (s, AV_LOG_DEBUG, "  fronsay: %s\n", &text[i + 1]);

            }

#endif

            break;



        case AUDI_TAG:

            /* send out audio chunk */

            ret= av_get_packet(pb, pkt, size);

            pkt->stream_index = wc3->audio_stream_index;

            pkt->pts = wc3->pts;



            /* time to advance pts */

            wc3->pts++;



            packet_read = 1;

            break;



        default:

            av_log (s, AV_LOG_ERROR, "  unrecognized WC3 chunk: %c%c%c%c (0x%02X%02X%02X%02X)\n",

                (uint8_t)fourcc_tag, (uint8_t)(fourcc_tag >> 8), (uint8_t)(fourcc_tag >> 16), (uint8_t)(fourcc_tag >> 24),

                (uint8_t)fourcc_tag, (uint8_t)(fourcc_tag >> 8), (uint8_t)(fourcc_tag >> 16), (uint8_t)(fourcc_tag >> 24));

            ret = AVERROR_INVALIDDATA;

            packet_read = 1;

            break;

        }

    }



    return ret;

}
