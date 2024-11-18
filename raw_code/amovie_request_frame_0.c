static int amovie_request_frame(AVFilterLink *outlink)

{

    MovieContext *movie = outlink->src->priv;

    int ret;



    if (movie->is_done)

        return AVERROR_EOF;

    do {

        if ((ret = amovie_get_samples(outlink)) < 0)

            return ret;

    } while (!movie->samplesref);



    ff_filter_samples(outlink, avfilter_ref_buffer(movie->samplesref, ~0));

    avfilter_unref_buffer(movie->samplesref);

    movie->samplesref = NULL;



    return 0;

}
