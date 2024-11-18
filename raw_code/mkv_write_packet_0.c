static int mkv_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    MatroskaMuxContext *mkv = s->priv_data;

    int codec_type          = s->streams[pkt->stream_index]->codec->codec_type;

    int keyframe            = !!(pkt->flags & AV_PKT_FLAG_KEY);

    int cluster_size;

    int cluster_size_limit;

    int64_t cluster_time;

    int64_t cluster_time_limit;

    AVIOContext *pb;

    int ret;



    if (mkv->tracks[pkt->stream_index].write_dts)

        cluster_time = pkt->dts - mkv->cluster_pts;

    else

        cluster_time = pkt->pts - mkv->cluster_pts;



    // start a new cluster every 5 MB or 5 sec, or 32k / 1 sec for streaming or

    // after 4k and on a keyframe

    if (s->pb->seekable) {

        pb = s->pb;

        cluster_size = avio_tell(pb) - mkv->cluster_pos;

        cluster_time_limit = 5000;

        cluster_size_limit = 5 * 1024 * 1024;

    } else {

        pb = mkv->dyn_bc;

        cluster_size = avio_tell(pb);

        cluster_time_limit = 1000;

        cluster_size_limit = 32 * 1024;

    }



    if (mkv->cluster_pos &&

        (cluster_size > cluster_size_limit ||

         cluster_time > cluster_time_limit ||

         (codec_type == AVMEDIA_TYPE_VIDEO && keyframe &&

          cluster_size > 4 * 1024))) {

        av_log(s, AV_LOG_DEBUG, "Starting new cluster at offset %" PRIu64

               " bytes, pts %" PRIu64 "dts %" PRIu64 "\n",

               avio_tell(pb), pkt->pts, pkt->dts);

        end_ebml_master(pb, mkv->cluster);

        mkv->cluster_pos = 0;

        if (mkv->dyn_bc)

            mkv_flush_dynbuf(s);

    }



    // check if we have an audio packet cached

    if (mkv->cur_audio_pkt.size > 0) {

        ret = mkv_write_packet_internal(s, &mkv->cur_audio_pkt);

        av_free_packet(&mkv->cur_audio_pkt);

        if (ret < 0) {

            av_log(s, AV_LOG_ERROR, "Could not write cached audio packet ret:%d\n", ret);

            return ret;

        }

    }



    // buffer an audio packet to ensure the packet containing the video

    // keyframe's timecode is contained in the same cluster for WebM

    if (codec_type == AVMEDIA_TYPE_AUDIO) {

        mkv->cur_audio_pkt = *pkt;

        if (pkt->buf) {

            mkv->cur_audio_pkt.buf = av_buffer_ref(pkt->buf);

            ret = mkv->cur_audio_pkt.buf ? 0 : AVERROR(ENOMEM);

        } else

            ret = av_dup_packet(&mkv->cur_audio_pkt);

    } else

        ret = mkv_write_packet_internal(s, pkt);

    return ret;

}
