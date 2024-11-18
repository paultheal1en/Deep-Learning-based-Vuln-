vorbis_comment(AVFormatContext * as, uint8_t *buf, int size)

{

    const uint8_t *p = buf;

    const uint8_t *end = buf + size;

    unsigned s, n, j;



    if (size < 8) /* must have vendor_length and user_comment_list_length */

        return -1;



    s = bytestream_get_le32(&p);



    if (end - p < s)

        return -1;



    p += s;



    n = bytestream_get_le32(&p);



    while (p < end && n > 0) {

        const char *t, *v;

        int tl, vl;



        s = bytestream_get_le32(&p);



        if (end - p < s)

            break;



        t = p;

        p += s;

        n--;



        v = memchr(t, '=', s);

        if (!v)

            continue;



        tl = v - t;

        vl = s - tl - 1;

        v++;



        if (tl && vl) {

            char *tt, *ct;



            tt = av_malloc(tl + 1);

            ct = av_malloc(vl + 1);

            if (!tt || !ct) {

                av_freep(&tt);

                av_freep(&ct);

                av_log(as, AV_LOG_WARNING, "out-of-memory error. skipping VorbisComment tag.\n");

                continue;

            }



            for (j = 0; j < tl; j++)

                tt[j] = toupper(t[j]);

            tt[tl] = 0;



            memcpy(ct, v, vl);

            ct[vl] = 0;



            av_metadata_set(&as->metadata, tt, ct);



            av_freep(&tt);

            av_freep(&ct);

        }

    }



    if (p != end)

        av_log(as, AV_LOG_INFO, "%ti bytes of comment header remain\n", end-p);

    if (n > 0)

        av_log(as, AV_LOG_INFO,

               "truncated comment header, %i comments not found\n", n);



    return 0;

}
