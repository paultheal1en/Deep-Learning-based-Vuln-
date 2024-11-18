static int cin_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    CinDemuxContext *cin = s->priv_data;

    ByteIOContext *pb = s->pb;

    CinFrameHeader *hdr = &cin->frame_header;

    int rc, palette_type, pkt_size;



    if (cin->audio_buffer_size == 0) {

        rc = cin_read_frame_header(cin, pb);

        if (rc)

            return rc;



        if ((int16_t)hdr->pal_colors_count < 0) {

            hdr->pal_colors_count = -(int16_t)hdr->pal_colors_count;

            palette_type = 1;

        } else {

            palette_type = 0;

        }



        /* palette and video packet */

        pkt_size = (palette_type + 3) * hdr->pal_colors_count + hdr->video_frame_size;



        if (av_new_packet(pkt, 4 + pkt_size))

            return AVERROR(ENOMEM);



        pkt->stream_index = cin->video_stream_index;

        pkt->pts = cin->video_stream_pts++;



        pkt->data[0] = palette_type;

        pkt->data[1] = hdr->pal_colors_count & 0xFF;

        pkt->data[2] = hdr->pal_colors_count >> 8;

        pkt->data[3] = hdr->video_frame_type;



        if (get_buffer(pb, &pkt->data[4], pkt_size) != pkt_size)

            return AVERROR(EIO);



        /* sound buffer will be processed on next read_packet() call */

        cin->audio_buffer_size = hdr->audio_frame_size;

        return 0;

    }



    /* audio packet */

    if (av_new_packet(pkt, cin->audio_buffer_size))

        return AVERROR(ENOMEM);



    pkt->stream_index = cin->audio_stream_index;

    pkt->pts = cin->audio_stream_pts;

    cin->audio_stream_pts += cin->audio_buffer_size * 2 / cin->file_header.audio_frame_size;



    if (get_buffer(pb, pkt->data, cin->audio_buffer_size) != cin->audio_buffer_size)

        return AVERROR(EIO);



    cin->audio_buffer_size = 0;

    return 0;

}
