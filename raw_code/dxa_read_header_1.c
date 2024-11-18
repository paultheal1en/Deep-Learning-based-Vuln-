static int dxa_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    AVIOContext *pb = s->pb;

    DXAContext *c = s->priv_data;

    AVStream *st, *ast;

    uint32_t tag;

    int32_t fps;

    int w, h;

    int num, den;

    int flags;



    tag = avio_rl32(pb);

    if (tag != MKTAG('D', 'E', 'X', 'A'))

        return -1;

    flags = avio_r8(pb);

    c->frames = avio_rb16(pb);

    if(!c->frames){

        av_log(s, AV_LOG_ERROR, "File contains no frames ???\n");

        return -1;

    }



    fps = avio_rb32(pb);

    if(fps > 0){

        den = 1000;

        num = fps;

    }else if (fps < 0){

        den = 100000;

        num = -fps;

    }else{

        den = 10;

        num = 1;

    }

    w = avio_rb16(pb);

    h = avio_rb16(pb);

    c->has_sound = 0;



    st = av_new_stream(s, 0);

    if (!st)

        return -1;



    // Parse WAV data header

    if(avio_rl32(pb) == MKTAG('W', 'A', 'V', 'E')){

        uint32_t size, fsize;

        c->has_sound = 1;

        size = avio_rb32(pb);

        c->vidpos = avio_tell(pb) + size;

        avio_skip(pb, 16);

        fsize = avio_rl32(pb);



        ast = av_new_stream(s, 0);

        if (!ast)

            return -1;

        ff_get_wav_header(pb, ast->codec, fsize);

        // find 'data' chunk

        while(avio_tell(pb) < c->vidpos && !pb->eof_reached){

            tag = avio_rl32(pb);

            fsize = avio_rl32(pb);

            if(tag == MKTAG('d', 'a', 't', 'a')) break;

            avio_skip(pb, fsize);

        }

        c->bpc = (fsize + c->frames - 1) / c->frames;

        if(ast->codec->block_align)

            c->bpc = ((c->bpc + ast->codec->block_align - 1) / ast->codec->block_align) * ast->codec->block_align;

        c->bytes_left = fsize;

        c->wavpos = avio_tell(pb);

        avio_seek(pb, c->vidpos, SEEK_SET);

    }



    /* now we are ready: build format streams */

    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id   = CODEC_ID_DXA;

    st->codec->width      = w;

    st->codec->height     = h;

    av_reduce(&den, &num, den, num, (1UL<<31)-1);

    av_set_pts_info(st, 33, num, den);

    /* flags & 0x80 means that image is interlaced,

     * flags & 0x40 means that image has double height

     * either way set true height

     */

    if(flags & 0xC0){

        st->codec->height >>= 1;

    }

    c->readvid = !c->has_sound;

    c->vidpos  = avio_tell(pb);

    s->start_time = 0;

    s->duration = (int64_t)c->frames * AV_TIME_BASE * num / den;

    av_log(s, AV_LOG_DEBUG, "%d frame(s)\n",c->frames);



    return 0;

}
