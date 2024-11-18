static int mxf_write_header_metadata_sets(AVFormatContext *s)

{

    AVStream *st;

    MXFStreamContext *sc = NULL;

    int i;



    mxf_write_preface(s);

    mxf_write_identification(s);

    mxf_write_content_storage(s);



    for (i = 0; i < s->nb_streams; i++) {

        st = s->streams[i];

        sc = av_mallocz(sizeof(MXFStreamContext));

        if (!sc)

            return AVERROR(ENOMEM);

        st->priv_data = sc;

        // set pts information

        if (st->codec->codec_type == CODEC_TYPE_VIDEO)

            av_set_pts_info(st, 64, 1, st->codec->time_base.den);

        else if (st->codec->codec_type == CODEC_TYPE_AUDIO)

            av_set_pts_info(st, 64, 1, st->codec->sample_rate);

    }



    mxf_build_structural_metadata(s, MaterialPackage);

    mxf_build_structural_metadata(s, SourcePackage);

    return 0;

}
