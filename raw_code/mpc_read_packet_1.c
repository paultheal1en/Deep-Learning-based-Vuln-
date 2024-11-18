static int mpc_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    MPCContext *c = s->priv_data;

    int ret, size, size2, curbits, cur = c->curframe;

    int64_t tmp, pos;



    if (c->curframe >= c->fcount)

        return -1;



    if(c->curframe != c->lastframe + 1){

        url_fseek(s->pb, c->frames[c->curframe].pos, SEEK_SET);

        c->curbits = c->frames[c->curframe].skip;

    }

    c->lastframe = c->curframe;

    c->curframe++;

    curbits = c->curbits;

    pos = url_ftell(s->pb);

    tmp = get_le32(s->pb);

    if(curbits <= 12){

        size2 = (tmp >> (12 - curbits)) & 0xFFFFF;

    }else{

        tmp = (tmp << 32) | get_le32(s->pb);

        size2 = (tmp >> (44 - curbits)) & 0xFFFFF;

    }

    curbits += 20;

    url_fseek(s->pb, pos, SEEK_SET);



    size = ((size2 + curbits + 31) & ~31) >> 3;

    if(cur == c->frames_noted){

        c->frames[cur].pos = pos;

        c->frames[cur].size = size;

        c->frames[cur].skip = curbits - 20;

        av_add_index_entry(s->streams[0], cur, cur, size, 0, AVINDEX_KEYFRAME);

        c->frames_noted++;

    }

    c->curbits = (curbits + size2) & 0x1F;



    if (av_new_packet(pkt, size) < 0)

        return AVERROR(EIO);



    pkt->data[0] = curbits;

    pkt->data[1] = (c->curframe > c->fcount);





    pkt->stream_index = 0;

    pkt->pts = cur;

    ret = get_buffer(s->pb, pkt->data + 4, size);

    if(c->curbits)

        url_fseek(s->pb, -4, SEEK_CUR);

    if(ret < size){

        av_free_packet(pkt);

        return AVERROR(EIO);

    }

    pkt->size = ret + 4;



    return 0;

}