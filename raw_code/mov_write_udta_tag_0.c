static int mov_write_udta_tag(AVIOContext *pb, MOVMuxContext *mov,

                              AVFormatContext *s)

{

    AVIOContext *pb_buf;

    int i, ret, size;

    uint8_t *buf;



    for (i = 0; i < s->nb_streams; i++)

        if (s->flags & AVFMT_FLAG_BITEXACT) {

            return 0;

        }



    ret = avio_open_dyn_buf(&pb_buf);

    if (ret < 0)

        return ret;



    if (mov->mode & MODE_3GP) {

        mov_write_3gp_udta_tag(pb_buf, s, "perf", "artist");

        mov_write_3gp_udta_tag(pb_buf, s, "titl", "title");

        mov_write_3gp_udta_tag(pb_buf, s, "auth", "author");

        mov_write_3gp_udta_tag(pb_buf, s, "gnre", "genre");

        mov_write_3gp_udta_tag(pb_buf, s, "dscp", "comment");

        mov_write_3gp_udta_tag(pb_buf, s, "albm", "album");

        mov_write_3gp_udta_tag(pb_buf, s, "cprt", "copyright");

        mov_write_3gp_udta_tag(pb_buf, s, "yrrc", "date");

    } else if (mov->mode == MODE_MOV) { // the title field breaks gtkpod with mp4 and my suspicion is that stuff is not valid in mp4

        mov_write_string_metadata(s, pb_buf, "\251ART", "artist",      0);

        mov_write_string_metadata(s, pb_buf, "\251nam", "title",       0);

        mov_write_string_metadata(s, pb_buf, "\251aut", "author",      0);

        mov_write_string_metadata(s, pb_buf, "\251alb", "album",       0);

        mov_write_string_metadata(s, pb_buf, "\251day", "date",        0);

        mov_write_string_metadata(s, pb_buf, "\251swr", "encoder",     0);

        mov_write_string_metadata(s, pb_buf, "\251des", "comment",     0);

        mov_write_string_metadata(s, pb_buf, "\251gen", "genre",       0);

        mov_write_string_metadata(s, pb_buf, "\251cpy", "copyright",   0);

    } else {

        /* iTunes meta data */

        mov_write_meta_tag(pb_buf, mov, s);

    }



    if (s->nb_chapters && !(mov->flags & FF_MOV_FLAG_DISABLE_CHPL))

        mov_write_chpl_tag(pb_buf, s);



    if ((size = avio_close_dyn_buf(pb_buf, &buf)) > 0) {

        avio_wb32(pb, size + 8);

        ffio_wfourcc(pb, "udta");

        avio_write(pb, buf, size);

    }

    av_free(buf);



    return 0;

}
