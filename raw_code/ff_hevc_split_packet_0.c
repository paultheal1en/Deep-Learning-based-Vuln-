int ff_hevc_split_packet(HEVCContext *s, HEVCPacket *pkt, const uint8_t *buf, int length,

                         AVCodecContext *avctx, int is_nalff, int nal_length_size)

{

    int consumed, ret = 0;



    pkt->nb_nals = 0;

    while (length >= 4) {

        HEVCNAL *nal;

        int extract_length = 0;



        if (is_nalff) {

            int i;

            for (i = 0; i < nal_length_size; i++)

                extract_length = (extract_length << 8) | buf[i];

            buf    += nal_length_size;

            length -= nal_length_size;



            if (extract_length > length) {

                av_log(avctx, AV_LOG_ERROR, "Invalid NAL unit size.\n");

                return AVERROR_INVALIDDATA;

            }

        } else {

            /* search start code */

            while (buf[0] != 0 || buf[1] != 0 || buf[2] != 1) {

                ++buf;

                --length;

                if (length < 4) {

                    if (pkt->nb_nals > 0) {

                        // No more start codes: we discarded some irrelevant

                        // bytes at the end of the packet.

                        return 0;

                    } else {

                        av_log(avctx, AV_LOG_ERROR, "No start code is found.\n");

                        return AVERROR_INVALIDDATA;

                    }

                }

            }



            buf           += 3;

            length        -= 3;

            extract_length = length;

        }



        if (pkt->nals_allocated < pkt->nb_nals + 1) {

            int new_size = pkt->nals_allocated + 1;

            void *tmp = av_realloc_array(pkt->nals, new_size, sizeof(*pkt->nals));



            if (!tmp)

                return AVERROR(ENOMEM);



            pkt->nals = tmp;

            memset(pkt->nals + pkt->nals_allocated, 0,

                   (new_size - pkt->nals_allocated) * sizeof(*pkt->nals));



            nal = &pkt->nals[pkt->nb_nals];

            nal->skipped_bytes_pos_size = 1024; // initial buffer size

            nal->skipped_bytes_pos = av_malloc_array(nal->skipped_bytes_pos_size, sizeof(*nal->skipped_bytes_pos));

            if (!nal->skipped_bytes_pos)

                return AVERROR(ENOMEM);



            pkt->nals_allocated = new_size;

        }

        nal = &pkt->nals[pkt->nb_nals];



        consumed = ff_hevc_extract_rbsp(s, buf, extract_length, nal);

        if (consumed < 0)

            return consumed;



        pkt->nb_nals++;



        ret = init_get_bits8(&nal->gb, nal->data, nal->size);

        if (ret < 0)

            return ret;



        ret = hls_nal_unit(nal, avctx);

        if (ret <= 0) {

            if (ret < 0) {

                av_log(avctx, AV_LOG_ERROR, "Invalid NAL unit %d, skipping.\n",

                       nal->type);

            }

            pkt->nb_nals--;

        }



        buf    += consumed;

        length -= consumed;

    }



    return 0;

}
