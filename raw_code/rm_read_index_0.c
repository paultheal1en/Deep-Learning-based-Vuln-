static int rm_read_index(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    unsigned int size, n_pkts, str_id, next_off, n, pos, pts;

    AVStream *st;



    do {

        if (avio_rl32(pb) != MKTAG('I','N','D','X'))

            return -1;

        size     = avio_rb32(pb);

        if (size < 20)

            return -1;

        avio_skip(pb, 2);

        n_pkts   = avio_rb32(pb);

        str_id   = avio_rb16(pb);

        next_off = avio_rb32(pb);

        for (n = 0; n < s->nb_streams; n++)

            if (s->streams[n]->id == str_id) {

                st = s->streams[n];

                break;

            }

        if (n == s->nb_streams)

            goto skip;



        for (n = 0; n < n_pkts; n++) {

            avio_skip(pb, 2);

            pts = avio_rb32(pb);

            pos = avio_rb32(pb);

            avio_skip(pb, 4); /* packet no. */



            av_add_index_entry(st, pos, pts, 0, 0, AVINDEX_KEYFRAME);

        }



skip:

        if (next_off && avio_tell(pb) != next_off &&

            avio_seek(pb, next_off, SEEK_SET) < 0)

            return -1;

    } while (next_off);



    return 0;

}
