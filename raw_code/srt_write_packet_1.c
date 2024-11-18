static int srt_write_packet(AVFormatContext *avf, AVPacket *pkt)

{

    SRTContext *srt = avf->priv_data;

    int write_ts = avf->streams[0]->codec->codec_id != AV_CODEC_ID_SRT;



    srt->index++;

    if (write_ts) {

        int64_t s = pkt->pts, e, d = pkt->duration;



        if (d <= 0)

            /* For backward compatibility, fallback to convergence_duration. */

            d = pkt->convergence_duration;

        if (s == AV_NOPTS_VALUE || d < 0) {

            av_log(avf, AV_LOG_ERROR, "Insufficient timestamps.\n");

            return AVERROR(EINVAL);

        }

        e = s + d;

        avio_printf(avf->pb, "%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n",

                       srt->index,

                       (int)(s / 3600000),      (int)(s / 60000) % 60,

                       (int)(s /    1000) % 60, (int)(s %  1000),

                       (int)(e / 3600000),      (int)(e / 60000) % 60,

                       (int)(e /    1000) % 60, (int)(e %  1000));

    }

    avio_write(avf->pb, pkt->data, pkt->size);

    if (write_ts)

        avio_write(avf->pb, "\n\n", 2);

    avio_flush(avf->pb);

    return 0;

}
