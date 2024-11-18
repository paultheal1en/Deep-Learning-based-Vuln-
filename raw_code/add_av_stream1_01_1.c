static AVStream *add_av_stream1(FFServerStream *stream,

                                AVCodecContext *codec, int copy)

{

    AVStream *fst;



    if(stream->nb_streams >= FF_ARRAY_ELEMS(stream->streams))

        return NULL;



    fst = av_mallocz(sizeof(AVStream));

    if (!fst)

        return NULL;

    if (copy) {

        fst->codec = avcodec_alloc_context3(codec->codec);

        if (!fst->codec) {

            av_free(fst);

            return NULL;

        }

        avcodec_copy_context(fst->codec, codec);

    } else

        /* live streams must use the actual feed's codec since it may be

         * updated later to carry extradata needed by them.

         */

        fst->codec = codec;



    fst->priv_data = av_mallocz(sizeof(FeedData));


    fst->index = stream->nb_streams;

    avpriv_set_pts_info(fst, 33, 1, 90000);

    fst->sample_aspect_ratio = codec->sample_aspect_ratio;

    stream->streams[stream->nb_streams++] = fst;

    return fst;

}