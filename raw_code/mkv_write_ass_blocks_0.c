static int mkv_write_ass_blocks(AVFormatContext *s, AVIOContext *pb,

                                AVPacket *pkt)

{

    MatroskaMuxContext *mkv = s->priv_data;

    int i, layer = 0, max_duration = 0, size, line_size, data_size = pkt->size;

    uint8_t *start, *end, *data = pkt->data;

    ebml_master blockgroup;

    char buffer[2048];



    while (data_size) {

        int duration = ass_get_duration(data);

        max_duration = FFMAX(duration, max_duration);

        end          = memchr(data, '\n', data_size);

        size         = line_size = end ? end - data + 1 : data_size;

        size        -= end ? (end[-1] == '\r') + 1 : 0;

        start        = data;

        for (i = 0; i < 3; i++, start++)

            if (!(start = memchr(start, ',', size - (start - data))))

                return max_duration;

        size -= start - data;

        sscanf(data, "Dialogue: %d,", &layer);

        i = snprintf(buffer, sizeof(buffer), "%" PRId64 ",%d,",

                     s->streams[pkt->stream_index]->nb_frames, layer);

        size = FFMIN(i + size, sizeof(buffer));

        memcpy(buffer + i, start, size - i);



        av_log(s, AV_LOG_DEBUG,

               "Writing block at offset %" PRIu64 ", size %d, "

               "pts %" PRId64 ", duration %d\n",

               avio_tell(pb), size, pkt->pts, duration);

        blockgroup = start_ebml_master(pb, MATROSKA_ID_BLOCKGROUP,

                                       mkv_blockgroup_size(size));

        put_ebml_id(pb, MATROSKA_ID_BLOCK);

        put_ebml_num(pb, size + 4, 0);

        // this assumes stream_index is less than 126

        avio_w8(pb, 0x80 | (pkt->stream_index + 1));

        avio_wb16(pb, pkt->pts - mkv->cluster_pts);

        avio_w8(pb, 0);

        avio_write(pb, buffer, size);

        put_ebml_uint(pb, MATROSKA_ID_BLOCKDURATION, duration);

        end_ebml_master(pb, blockgroup);



        data      += line_size;

        data_size -= line_size;

    }



    return max_duration;

}
