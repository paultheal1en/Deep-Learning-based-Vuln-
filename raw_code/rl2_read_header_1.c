static av_cold int rl2_read_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    AVStream *st;

    unsigned int frame_count;

    unsigned int audio_frame_counter = 0;

    unsigned int video_frame_counter = 0;

    unsigned int back_size;

    unsigned short sound_rate;

    unsigned short rate;

    unsigned short channels;

    unsigned short def_sound_size;

    unsigned int signature;

    unsigned int pts_den = 11025; /* video only case */

    unsigned int pts_num = 1103;

    unsigned int* chunk_offset = NULL;

    int* chunk_size = NULL;

    int* audio_size = NULL;

    int i;

    int ret = 0;



    avio_skip(pb,4);          /* skip FORM tag */

    back_size = avio_rl32(pb); /**< get size of the background frame */

    signature = avio_rb32(pb);

    avio_skip(pb, 4);         /* data size */

    frame_count = avio_rl32(pb);



    /* disallow back_sizes and frame_counts that may lead to overflows later */

    if(back_size > INT_MAX/2  || frame_count > INT_MAX / sizeof(uint32_t))




    avio_skip(pb, 2);         /* encoding mentod */

    sound_rate = avio_rl16(pb);

    rate = avio_rl16(pb);

    channels = avio_rl16(pb);

    def_sound_size = avio_rl16(pb);







    /** setup video stream */

    st = avformat_new_stream(s, NULL);

    if(!st)

         return AVERROR(ENOMEM);



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = AV_CODEC_ID_RL2;

    st->codec->codec_tag = 0;  /* no fourcc */

    st->codec->width = 320;

    st->codec->height = 200;



    /** allocate and fill extradata */

    st->codec->extradata_size = EXTRADATA1_SIZE;



    if(signature == RLV3_TAG && back_size > 0)

        st->codec->extradata_size += back_size;



    st->codec->extradata = av_mallocz(st->codec->extradata_size +

                                          FF_INPUT_BUFFER_PADDING_SIZE);

    if(!st->codec->extradata)

        return AVERROR(ENOMEM);



    if(avio_read(pb,st->codec->extradata,st->codec->extradata_size) !=

                      st->codec->extradata_size)

        return AVERROR(EIO);



    /** setup audio stream if present */

    if(sound_rate){

        pts_num = def_sound_size;

        pts_den = rate;



        st = avformat_new_stream(s, NULL);

        if (!st)

            return AVERROR(ENOMEM);

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codec->codec_id = AV_CODEC_ID_PCM_U8;

        st->codec->codec_tag = 1;

        st->codec->channels = channels;

        st->codec->bits_per_coded_sample = 8;

        st->codec->sample_rate = rate;

        st->codec->bit_rate = st->codec->channels * st->codec->sample_rate *

            st->codec->bits_per_coded_sample;

        st->codec->block_align = st->codec->channels *

            st->codec->bits_per_coded_sample / 8;

        avpriv_set_pts_info(st,32,1,rate);




    avpriv_set_pts_info(s->streams[0], 32, pts_num, pts_den);



    chunk_size =   av_malloc(frame_count * sizeof(uint32_t));

    audio_size =   av_malloc(frame_count * sizeof(uint32_t));

    chunk_offset = av_malloc(frame_count * sizeof(uint32_t));



    if(!chunk_size || !audio_size || !chunk_offset){

        av_free(chunk_size);

        av_free(audio_size);

        av_free(chunk_offset);

        return AVERROR(ENOMEM);




    /** read offset and size tables */

    for(i=0; i < frame_count;i++)

        chunk_size[i] = avio_rl32(pb);

    for(i=0; i < frame_count;i++)

        chunk_offset[i] = avio_rl32(pb);

    for(i=0; i < frame_count;i++)

        audio_size[i] = avio_rl32(pb) & 0xFFFF;



    /** build the sample index */

    for(i=0;i<frame_count;i++){

        if(chunk_size[i] < 0 || audio_size[i] > chunk_size[i]){

            ret = AVERROR_INVALIDDATA;

            break;




        if(sound_rate && audio_size[i]){

            av_add_index_entry(s->streams[1], chunk_offset[i],

                audio_frame_counter,audio_size[i], 0, AVINDEX_KEYFRAME);

            audio_frame_counter += audio_size[i] / channels;


        av_add_index_entry(s->streams[0], chunk_offset[i] + audio_size[i],

            video_frame_counter,chunk_size[i]-audio_size[i],0,AVINDEX_KEYFRAME);

        ++video_frame_counter;






    av_free(chunk_size);

    av_free(audio_size);

    av_free(chunk_offset);



    return ret;
