static int xiph_handle_packet(AVFormatContext *ctx, PayloadContext *data,

                              AVStream *st, AVPacket *pkt, uint32_t *timestamp,

                              const uint8_t *buf, int len, uint16_t seq,

                              int flags)

{



    int ident, fragmented, tdt, num_pkts, pkt_len;



    if (!buf) {

        if (!data->split_buf || data->split_pos + 2 > data->split_buf_len ||

            data->split_pkts <= 0) {

            av_log(ctx, AV_LOG_ERROR, "No more data to return\n");

            return AVERROR_INVALIDDATA;

        }

        pkt_len = AV_RB16(data->split_buf + data->split_pos);

        data->split_pos += 2;

        if (data->split_pos + pkt_len > data->split_buf_len) {

            av_log(ctx, AV_LOG_ERROR, "Not enough data to return\n");

            return AVERROR_INVALIDDATA;

        }

        if (av_new_packet(pkt, pkt_len)) {

            av_log(ctx, AV_LOG_ERROR, "Out of memory.\n");

            return AVERROR(ENOMEM);

        }

        pkt->stream_index = st->index;

        memcpy(pkt->data, data->split_buf + data->split_pos, pkt_len);

        data->split_pos += pkt_len;

        data->split_pkts--;

        return data->split_pkts > 0;

    }



    if (len < 6 || len > INT_MAX/2) {

        av_log(ctx, AV_LOG_ERROR, "Invalid %d byte packet\n", len);

        return AVERROR_INVALIDDATA;

    }



    // read xiph rtp headers

    ident       = AV_RB24(buf);

    fragmented  = buf[3] >> 6;

    tdt         = (buf[3] >> 4) & 3;

    num_pkts    = buf[3] & 0xf;

    pkt_len     = AV_RB16(buf + 4);



    if (pkt_len > len - 6) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid packet length %d in %d byte packet\n", pkt_len,

               len);

        return AVERROR_INVALIDDATA;

    }



    if (ident != data->ident) {

        av_log(ctx, AV_LOG_ERROR,

               "Unimplemented Xiph SDP configuration change detected\n");

        return AVERROR_PATCHWELCOME;

    }



    if (tdt) {

        av_log(ctx, AV_LOG_ERROR,

               "Unimplemented RTP Xiph packet settings (%d,%d,%d)\n",

               fragmented, tdt, num_pkts);

        return AVERROR_PATCHWELCOME;

    }



    buf += 6; // move past header bits

    len -= 6;



    if (fragmented == 0) {

        if (av_new_packet(pkt, pkt_len)) {

            av_log(ctx, AV_LOG_ERROR, "Out of memory.\n");

            return AVERROR(ENOMEM);

        }

        pkt->stream_index = st->index;

        memcpy(pkt->data, buf, pkt_len);

        buf += pkt_len;

        len -= pkt_len;

        num_pkts--;



        if (num_pkts > 0) {

            if (len > data->split_buf_size || !data->split_buf) {

                av_freep(&data->split_buf);

                data->split_buf_size = 2 * len;

                data->split_buf = av_malloc(data->split_buf_size);

                if (!data->split_buf) {

                    av_log(ctx, AV_LOG_ERROR, "Out of memory.\n");

                    av_free_packet(pkt);

                    return AVERROR(ENOMEM);

                }

            }

            memcpy(data->split_buf, buf, len);

            data->split_buf_len = len;

            data->split_pos = 0;

            data->split_pkts = num_pkts;

            return 1;

        }



        return 0;



    } else if (fragmented == 1) {

        // start of xiph data fragment

        int res;



        // end packet has been lost somewhere, so drop buffered data

        ffio_free_dyn_buf(&data->fragment);



        if((res = avio_open_dyn_buf(&data->fragment)) < 0)

            return res;



        avio_write(data->fragment, buf, pkt_len);

        data->timestamp = *timestamp;



    } else {

        av_assert1(fragmented < 4);

        if (data->timestamp != *timestamp) {

            // skip if fragmented timestamp is incorrect;

            // a start packet has been lost somewhere

            ffio_free_dyn_buf(&data->fragment);

            av_log(ctx, AV_LOG_ERROR, "RTP timestamps don't match!\n");

            return AVERROR_INVALIDDATA;

        }

        if (!data->fragment) {

            av_log(ctx, AV_LOG_WARNING,

                   "Received packet without a start fragment; dropping.\n");

            return AVERROR(EAGAIN);

        }



        // copy data to fragment buffer

        avio_write(data->fragment, buf, pkt_len);



        if (fragmented == 3) {

            // end of xiph data packet

            int ret = ff_rtp_finalize_packet(pkt, &data->fragment, st->index);

            if (ret < 0) {

                av_log(ctx, AV_LOG_ERROR,

                       "Error occurred when getting fragment buffer.");

                return ret;

            }



            return 0;

        }

    }



   return AVERROR(EAGAIN);

}
