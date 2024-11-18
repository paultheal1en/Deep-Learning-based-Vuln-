static int ogg_read_page(AVFormatContext *s, int *str)

{

    AVIOContext *bc = s->pb;

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os;

    int ret, i = 0;

    int flags, nsegs;

    uint64_t gp;

    uint32_t serial;

    int size, idx;

    uint8_t sync[4];

    int sp = 0;



    ret = avio_read(bc, sync, 4);

    if (ret < 4)

        return ret < 0 ? ret : AVERROR_EOF;



    do{

        int c;



        if (sync[sp & 3] == 'O' &&

            sync[(sp + 1) & 3] == 'g' &&

            sync[(sp + 2) & 3] == 'g' && sync[(sp + 3) & 3] == 'S')

            break;



        c = avio_r8(bc);

        if (bc->eof_reached)

            return AVERROR_EOF;

        sync[sp++ & 3] = c;

    }while (i++ < MAX_PAGE_SIZE);



    if (i >= MAX_PAGE_SIZE){

        av_log (s, AV_LOG_INFO, "ogg, can't find sync word\n");

        return AVERROR_INVALIDDATA;

    }



    if (avio_r8(bc) != 0)      /* version */

        return AVERROR_INVALIDDATA;



    flags = avio_r8(bc);

    gp = avio_rl64 (bc);

    serial = avio_rl32 (bc);

    avio_skip(bc, 8); /* seq, crc */

    nsegs = avio_r8(bc);



    idx = ogg_find_stream (ogg, serial);

    if (idx < 0){

        if (ogg->headers) {

            int n;



            for (n = 0; n < ogg->nstreams; n++) {

                av_freep(&ogg->streams[n].buf);

                if (!ogg->state || ogg->state->streams[n].private != ogg->streams[n].private)

                    av_freep(&ogg->streams[n].private);

            }

            ogg->curidx   = -1;

            ogg->nstreams = 0;

            idx = ogg_new_stream(s, serial, 0);

        } else {

            idx = ogg_new_stream(s, serial, 1);

        }

        if (idx < 0)

            return idx;

    }



    os = ogg->streams + idx;

    os->page_pos = avio_tell(bc) - 27;



    if(os->psize > 0)

        ogg_new_buf(ogg, idx);



    ret = avio_read(bc, os->segments, nsegs);

    if (ret < nsegs)

        return ret < 0 ? ret : AVERROR_EOF;



    os->nsegs = nsegs;

    os->segp = 0;



    size = 0;

    for (i = 0; i < nsegs; i++)

        size += os->segments[i];



    if (flags & OGG_FLAG_CONT || os->incomplete){

        if (!os->psize){

            while (os->segp < os->nsegs){

                int seg = os->segments[os->segp++];

                os->pstart += seg;

                if (seg < 255)

                    break;

            }

            os->sync_pos = os->page_pos;

        }

    }else{

        os->psize = 0;

        os->sync_pos = os->page_pos;

    }



    if (os->bufsize - os->bufpos < size){

        uint8_t *nb = av_malloc (os->bufsize *= 2);

        memcpy (nb, os->buf, os->bufpos);

        av_free (os->buf);

        os->buf = nb;

    }



    ret = avio_read(bc, os->buf + os->bufpos, size);

    if (ret < size)

        return ret < 0 ? ret : AVERROR_EOF;



    os->bufpos += size;

    os->granule = gp;

    os->flags = flags;



    if (str)

        *str = idx;



    return 0;

}
