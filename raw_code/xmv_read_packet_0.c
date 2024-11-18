static int xmv_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    XMVDemuxContext *xmv = s->priv_data;

    int result;



    if (xmv->video.current_frame == xmv->video.frame_count) {

        /* No frames left in this packet, so we fetch a new one */



        result = xmv_fetch_new_packet(s);

        if (result)

            return result;

    }



    if (xmv->current_stream == 0) {

        /* Fetch a video frame */



        result = xmv_fetch_video_packet(s, pkt);

    } else {

        /* Fetch an audio frame */



        result = xmv_fetch_audio_packet(s, pkt, xmv->current_stream - 1);

    }

    if (result)

        return result;





    /* Increase our counters */

    if (++xmv->current_stream >= xmv->stream_count) {

        xmv->current_stream       = 0;

        xmv->video.current_frame += 1;

    }



    return 0;

}
