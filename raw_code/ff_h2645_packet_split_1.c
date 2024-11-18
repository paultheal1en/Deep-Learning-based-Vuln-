int ff_h2645_packet_split(H2645Packet *pkt, const uint8_t *buf, int length,

                          void *logctx, int is_nalff, int nal_length_size,

                          enum AVCodecID codec_id)

{

    int consumed, ret = 0;

    const uint8_t *next_avc = buf + (is_nalff ? 0 : length);



    pkt->nb_nals = 0;

    while (length >= 4) {

        H2645NAL *nal;

        int extract_length = 0;

        int skip_trailing_zeros = 1;



        /*

         * Only parse an AVC1 length field if one is expected at the current

         * buffer position. There are unfortunately streams with multiple

         * NAL units covered by the length field. Those NAL units are delimited

         * by Annex B start code prefixes. ff_h2645_extract_rbsp() detects it

         * correctly and consumes only the first NAL unit. The additional NAL

         * units are handled here in the Annex B parsing code.

         */

        if (buf == next_avc) {

            int i;

            for (i = 0; i < nal_length_size; i++)

                extract_length = (extract_length << 8) | buf[i];



            if (extract_length > length) {

                av_log(logctx, AV_LOG_ERROR,

                       "Invalid NAL unit size (%d > %d).\n",

                       extract_length, length);

                return AVERROR_INVALIDDATA;

            }

            buf     += nal_length_size;

            length  -= nal_length_size;

            // keep track of the next AVC1 length field

            next_avc = buf + extract_length;

        } else {

            /*

             * expected to return immediately except for streams with mixed

             * NAL unit coding

             */

            int buf_index = find_next_start_code(buf, next_avc);



            buf    += buf_index;

            length -= buf_index;



            /*

             * break if an AVC1 length field is expected at the current buffer

             * position

             */

            if (buf == next_avc)

                continue;



            if (length > 0) {

                extract_length = length;

            } else if (pkt->nb_nals == 0) {

                av_log(logctx, AV_LOG_ERROR, "No NAL unit found\n");

                return AVERROR_INVALIDDATA;

            } else {

                break;

            }

        }



        if (pkt->nals_allocated < pkt->nb_nals + 1) {

            int new_size = pkt->nals_allocated + 1;

            H2645NAL *tmp = av_realloc_array(pkt->nals, new_size, sizeof(*tmp));

            if (!tmp)

                return AVERROR(ENOMEM);



            pkt->nals = tmp;

            memset(pkt->nals + pkt->nals_allocated, 0,

                   (new_size - pkt->nals_allocated) * sizeof(*tmp));

            pkt->nals_allocated = new_size;

        }

        nal = &pkt->nals[pkt->nb_nals++];



        consumed = ff_h2645_extract_rbsp(buf, extract_length, nal);

        if (consumed < 0)

            return consumed;



        /* see commit 3566042a0 */

        if (consumed < length - 3 &&

            buf[consumed]     == 0x00 && buf[consumed + 1] == 0x00 &&

            buf[consumed + 2] == 0x01 && buf[consumed + 3] == 0xE0)

            skip_trailing_zeros = 0;



        nal->size_bits = get_bit_length(nal, skip_trailing_zeros);



        ret = init_get_bits(&nal->gb, nal->data, nal->size_bits);

        if (ret < 0)

            return ret;



        if (codec_id == AV_CODEC_ID_HEVC)

            ret = hevc_parse_nal_header(nal, logctx);

        else

            ret = h264_parse_nal_header(nal, logctx);

        if (ret <= 0) {

            if (ret < 0) {

                av_log(logctx, AV_LOG_ERROR, "Invalid NAL unit %d, skipping.\n",

                       nal->type);

            }

            pkt->nb_nals--;

        }



        buf    += consumed;

        length -= consumed;

    }



    return 0;

}
