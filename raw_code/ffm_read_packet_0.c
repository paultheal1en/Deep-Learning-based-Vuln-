static int ffm_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int size;

    FFMContext *ffm = s->priv_data;

    int duration, ret;



    switch(ffm->read_state) {

    case READ_HEADER:

        if ((ret = ffm_is_avail_data(s, FRAME_HEADER_SIZE+4)) < 0)

            return ret;



        av_dlog(s, "pos=%08"PRIx64" spos=%"PRIx64", write_index=%"PRIx64" size=%"PRIx64"\n",

               avio_tell(s->pb), s->pb->pos, ffm->write_index, ffm->file_size);

        if (ffm_read_data(s, ffm->header, FRAME_HEADER_SIZE, 1) !=

            FRAME_HEADER_SIZE)

            return -1;

        if (ffm->header[1] & FLAG_DTS)

            if (ffm_read_data(s, ffm->header+16, 4, 1) != 4)

                return -1;

        ffm->read_state = READ_DATA;

        /* fall thru */

    case READ_DATA:

        size = AV_RB24(ffm->header + 2);

        if ((ret = ffm_is_avail_data(s, size)) < 0)

            return ret;



        duration = AV_RB24(ffm->header + 5);



        av_new_packet(pkt, size);

        pkt->stream_index = ffm->header[0];

        if ((unsigned)pkt->stream_index >= s->nb_streams) {

            av_log(s, AV_LOG_ERROR, "invalid stream index %d\n", pkt->stream_index);

            av_free_packet(pkt);

            ffm->read_state = READ_HEADER;

            return -1;

        }

        pkt->pos = avio_tell(s->pb);

        if (ffm->header[1] & FLAG_KEY_FRAME)

            pkt->flags |= AV_PKT_FLAG_KEY;



        ffm->read_state = READ_HEADER;

        if (ffm_read_data(s, pkt->data, size, 0) != size) {

            /* bad case: desynchronized packet. we cancel all the packet loading */

            av_free_packet(pkt);

            return -1;

        }

        pkt->pts = AV_RB64(ffm->header+8);

        if (ffm->header[1] & FLAG_DTS)

            pkt->dts = pkt->pts - AV_RB32(ffm->header+16);

        else

            pkt->dts = pkt->pts;

        pkt->duration = duration;

        break;

    }

    return 0;

}
