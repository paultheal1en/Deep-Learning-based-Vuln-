static int mpc8_read_header(AVFormatContext *s)

{

    MPCContext *c = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st;

    int tag = 0;

    int64_t size, pos;



    c->header_pos = avio_tell(pb);

    if(avio_rl32(pb) != TAG_MPCK){

        av_log(s, AV_LOG_ERROR, "Not a Musepack8 file\n");





    while(!avio_feof(pb)){

        pos = avio_tell(pb);

        mpc8_get_chunk_header(pb, &tag, &size);





        if(tag == TAG_STREAMHDR)

            break;

        mpc8_handle_chunk(s, tag, pos, size);


    if(tag != TAG_STREAMHDR){

        av_log(s, AV_LOG_ERROR, "Stream header not found\n");



    pos = avio_tell(pb);

    avio_skip(pb, 4); //CRC

    c->ver = avio_r8(pb);

    if(c->ver != 8){

        av_log(s, AV_LOG_ERROR, "Unknown stream version %d\n", c->ver);

        return AVERROR_PATCHWELCOME;


    c->samples = ffio_read_varlen(pb);

    ffio_read_varlen(pb); //silence samples at the beginning



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id = AV_CODEC_ID_MUSEPACK8;

    st->codec->bits_per_coded_sample = 16;



    if (ff_get_extradata(st->codec, pb, 2) < 0)

        return AVERROR(ENOMEM);



    st->codec->channels = (st->codec->extradata[1] >> 4) + 1;

    st->codec->sample_rate = mpc8_rate[st->codec->extradata[0] >> 5];

    avpriv_set_pts_info(st, 32, 1152  << (st->codec->extradata[1]&3)*2, st->codec->sample_rate);

    st->start_time = 0;

    st->duration = c->samples / (1152 << (st->codec->extradata[1]&3)*2);

    size -= avio_tell(pb) - pos;

    if (size > 0)

        avio_skip(pb, size);



    if (pb->seekable) {

        int64_t pos = avio_tell(s->pb);

        c->apetag_start = ff_ape_parse_tag(s);

        avio_seek(s->pb, pos, SEEK_SET);




    return 0;
