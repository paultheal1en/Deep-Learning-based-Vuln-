static int wc3_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    Wc3DemuxContext *wc3 = s->priv_data;

    ByteIOContext *pb = s->pb;

    unsigned int fourcc_tag;

    unsigned int size;

    int packet_read = 0;

    int ret = 0;

    unsigned char preamble[WC3_PREAMBLE_SIZE];

    unsigned char text[1024];

    unsigned int palette_number;

    int i;

    unsigned char r, g, b;

    int base_palette_index;



    while (!packet_read) {



        /* get the next chunk preamble */

        if ((ret = get_buffer(pb, preamble, WC3_PREAMBLE_SIZE)) !=

            WC3_PREAMBLE_SIZE)

            ret = AVERROR(EIO);



        fourcc_tag = AV_RL32(&preamble[0]);

        /* chunk sizes are 16-bit aligned */

        size = (AV_RB32(&preamble[4]) + 1) & (~1);



        switch (fourcc_tag) {



        case BRCH_TAG:

            /* no-op */

            break;



        case SHOT_TAG:

            /* load up new palette */

            if ((ret = get_buffer(pb, preamble, 4)) != 4)

                return AVERROR(EIO);

            palette_number = AV_RL32(&preamble[0]);

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

            if (ret != size)

                ret = AVERROR(EIO);

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

            if (ret != size)

                ret = AVERROR(EIO);



            /* time to advance pts */

            wc3->pts++;



            packet_read = 1;

            break;



        default:

            av_log (s, AV_LOG_ERROR, "  unrecognized WC3 chunk: %c%c%c%c (0x%02X%02X%02X%02X)\n",

                preamble[0], preamble[1], preamble[2], preamble[3],

                preamble[0], preamble[1], preamble[2], preamble[3]);

            ret = AVERROR_INVALIDDATA;

            packet_read = 1;

            break;

        }

    }



    return ret;

}
