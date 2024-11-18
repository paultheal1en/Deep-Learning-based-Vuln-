static int mp3_read_header(AVFormatContext *s)

{

    MP3DecContext *mp3 = s->priv_data;

    AVStream *st;

    int64_t off;

    int ret;

    int i;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id = AV_CODEC_ID_MP3;

    st->need_parsing = AVSTREAM_PARSE_FULL_RAW;

    st->start_time = 0;



    // lcm of all mp3 sample rates

    avpriv_set_pts_info(st, 64, 1, 14112000);



    s->pb->maxsize = -1;

    off = avio_tell(s->pb);



    if (!av_dict_get(s->metadata, "", NULL, AV_DICT_IGNORE_SUFFIX))

        ff_id3v1_read(s);



    if(s->pb->seekable)

        mp3->filesize = avio_size(s->pb);



    if (mp3_parse_vbr_tags(s, st, off) < 0)

        avio_seek(s->pb, off, SEEK_SET);



    ret = ff_replaygain_export(st, s->metadata);

    if (ret < 0)

        return ret;



    off = avio_tell(s->pb);

    for (i = 0; i < 64 * 1024; i++) {

        uint32_t header, header2;

        int frame_size;

        if (!(i&1023))

            ffio_ensure_seekback(s->pb, i + 1024 + 4);

        frame_size = check(s->pb, off + i, &header);

        if (frame_size > 0) {

            avio_seek(s->pb, off, SEEK_SET);

            ffio_ensure_seekback(s->pb, i + 1024 + frame_size + 4);

            if (check(s->pb, off + i + frame_size, &header2) >= 0 &&

                (header & SAME_HEADER_MASK) == (header2 & SAME_HEADER_MASK))

            {

                av_log(s, AV_LOG_INFO, "Skipping %d bytes of junk at %"PRId64".\n", i, off);

                avio_seek(s->pb, off + i, SEEK_SET);

                break;

            }

        }

        avio_seek(s->pb, off, SEEK_SET);

    }



    // the seek index is relative to the end of the xing vbr headers

    for (i = 0; i < st->nb_index_entries; i++)

        st->index_entries[i].pos += avio_tell(s->pb);



    /* the parameters will be extracted from the compressed bitstream */

    return 0;

}
