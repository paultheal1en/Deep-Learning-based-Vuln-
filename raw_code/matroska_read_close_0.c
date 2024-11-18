matroska_read_close (AVFormatContext *s)

{

    MatroskaDemuxContext *matroska = s->priv_data;

    int n = 0;



    if (matroska->writing_app)

        av_free(matroska->writing_app);

    if (matroska->muxing_app)

        av_free(matroska->muxing_app);

    if (matroska->index)

        av_free(matroska->index);



    if (matroska->packets != NULL) {

        for (n = 0; n < matroska->num_packets; n++) {

            av_free_packet(matroska->packets[n]);

            av_free(matroska->packets[n]);

        }

        av_free(matroska->packets);

    }



    for (n = 0; n < matroska->num_tracks; n++) {

        MatroskaTrack *track = matroska->tracks[n];

        if (track->codec_id)

            av_free(track->codec_id);

        if (track->codec_name)

            av_free(track->codec_name);

        if (track->codec_priv)

            av_free(track->codec_priv);

        if (track->name)

            av_free(track->name);

        if (track->language)

            av_free(track->language);



        av_free(track);

    }



    memset(matroska, 0, sizeof(MatroskaDemuxContext));



    return 0;

}
