static void mpc8_parse_seektable(AVFormatContext *s, int64_t off)

{

    MPCContext *c = s->priv_data;

    int tag;

    int64_t size, pos, ppos[2];

    uint8_t *buf;

    int i, t, seekd;

    GetBitContext gb;



    avio_seek(s->pb, off, SEEK_SET);

    mpc8_get_chunk_header(s->pb, &tag, &size);

    if(tag != TAG_SEEKTABLE){

        av_log(s, AV_LOG_ERROR, "No seek table at given position\n");







    if(!(buf = av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE)))


    avio_read(s->pb, buf, size);

    init_get_bits(&gb, buf, size * 8);

    size = gb_get_v(&gb);

    if(size > UINT_MAX/4 || size > c->samples/1152){

        av_log(s, AV_LOG_ERROR, "Seek table is too big\n");



    seekd = get_bits(&gb, 4);

    for(i = 0; i < 2; i++){

        pos = gb_get_v(&gb) + c->header_pos;

        ppos[1 - i] = pos;

        av_add_index_entry(s->streams[0], pos, i, 0, 0, AVINDEX_KEYFRAME);


    for(; i < size; i++){

        t = get_unary(&gb, 1, 33) << 12;

        t += get_bits(&gb, 12);

        if(t & 1)

            t = -(t & ~1);

        pos = (t >> 1) + ppos[0]*2 - ppos[1];

        av_add_index_entry(s->streams[0], pos, i << seekd, 0, 0, AVINDEX_KEYFRAME);

        ppos[1] = ppos[0];

        ppos[0] = pos;


    av_free(buf);
