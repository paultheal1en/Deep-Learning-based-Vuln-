static int yop_read_seek(AVFormatContext *s, int stream_index,

                         int64_t timestamp, int flags)

{

    YopDecContext *yop = s->priv_data;

    int64_t frame_pos, pos_min, pos_max;

    int frame_count;



    av_free_packet(&yop->video_packet);



    if (!stream_index)

        return -1;



    pos_min        = s->data_offset;

    pos_max        = avio_size(s->pb) - yop->frame_size;

    frame_count    = (pos_max - pos_min) / yop->frame_size;



    timestamp      = FFMAX(0, FFMIN(frame_count, timestamp));



    frame_pos      = timestamp * yop->frame_size + pos_min;

    yop->odd_frame = timestamp & 1;



    avio_seek(s->pb, frame_pos, SEEK_SET);

    return 0;

}
