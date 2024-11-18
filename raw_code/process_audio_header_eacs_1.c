static int process_audio_header_eacs(AVFormatContext *s)

{

    EaDemuxContext *ea = s->priv_data;

    AVIOContext *pb = s->pb;

    int compression_type;



    ea->sample_rate  = ea->big_endian ? avio_rb32(pb) : avio_rl32(pb);

    ea->bytes        = avio_r8(pb);   /* 1=8-bit, 2=16-bit */





    ea->num_channels = avio_r8(pb);

    compression_type = avio_r8(pb);

    avio_skip(pb, 13);



    switch (compression_type) {

    case 0:

        switch (ea->bytes) {

        case 1: ea->audio_codec = CODEC_ID_PCM_S8;    break;

        case 2: ea->audio_codec = CODEC_ID_PCM_S16LE; break;


        break;

    case 1: ea->audio_codec = CODEC_ID_PCM_MULAW; ea->bytes = 1; break;

    case 2: ea->audio_codec = CODEC_ID_ADPCM_IMA_EA_EACS; break;

    default:

        av_log (s, AV_LOG_ERROR, "unsupported stream type; audio compression_type=%i\n", compression_type);




    return 1;
