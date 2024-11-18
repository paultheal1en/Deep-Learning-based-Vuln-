static int read_header(AVFormatContext *s,

                       AVFormatParameters *ap)

{

    JVDemuxContext *jv = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *vst, *ast;

    int64_t audio_pts = 0;

    int64_t offset;

    int i;



    avio_skip(pb, 80);



    ast = av_new_stream(s, 0);

    vst = av_new_stream(s, 1);

    if (!ast || !vst)

        return AVERROR(ENOMEM);



    vst->codec->codec_type  = CODEC_TYPE_VIDEO;

    vst->codec->codec_id    = CODEC_ID_JV;

    vst->codec->codec_tag   = 0; /* no fourcc */

    vst->codec->width       = avio_rl16(pb);

    vst->codec->height      = avio_rl16(pb);

    vst->nb_frames          =

    ast->nb_index_entries   = avio_rl16(pb);

    av_set_pts_info(vst, 64, avio_rl16(pb), 1000);



    avio_skip(pb, 4);



    ast->codec->codec_type  = CODEC_TYPE_AUDIO;

    ast->codec->codec_id    = CODEC_ID_PCM_U8;

    ast->codec->codec_tag   = 0; /* no fourcc */

    ast->codec->sample_rate = avio_rl16(pb);

    ast->codec->channels    = 1;

    av_set_pts_info(ast, 64, 1, ast->codec->sample_rate);



    avio_skip(pb, 10);



    ast->index_entries = av_malloc(ast->nb_index_entries * sizeof(*ast->index_entries));

    if (!ast->index_entries)

        return AVERROR(ENOMEM);



    jv->frames = av_malloc(ast->nb_index_entries * sizeof(JVFrame));

    if (!jv->frames)

        return AVERROR(ENOMEM);



    offset = 0x68 + ast->nb_index_entries * 16;

    for(i = 0; i < ast->nb_index_entries; i++) {

        AVIndexEntry *e   = ast->index_entries + i;

        JVFrame      *jvf = jv->frames + i;



        /* total frame size including audio, video, palette data and padding */

        e->size         = avio_rl32(pb);

        e->timestamp    = i;

        e->pos          = offset;

        offset         += e->size;



        jvf->audio_size = avio_rl32(pb);

        jvf->video_size = avio_rl32(pb);

        jvf->palette_size = avio_r8(pb) ? 768 : 0;



        if (avio_r8(pb))

             av_log(s, AV_LOG_WARNING, "unsupported audio codec\n");

        jvf->video_type = avio_r8(pb);

        avio_skip(pb, 1);



        e->timestamp = jvf->audio_size ? audio_pts : AV_NOPTS_VALUE;

        audio_pts += jvf->audio_size;



        e->flags = jvf->video_type != 1 ? AVINDEX_KEYFRAME : 0;

    }



    jv->state = JV_AUDIO;

    return 0;

}