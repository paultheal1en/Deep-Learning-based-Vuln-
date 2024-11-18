static int ogg_buffer_data(AVFormatContext *s, AVStream *st,

                           uint8_t *data, unsigned size, int64_t granule,

                           int header)

{

    OGGStreamContext *oggstream = st->priv_data;

    OGGContext *ogg = s->priv_data;

    int total_segments = size / 255 + 1;

    uint8_t *p = data;

    int i, segments, len, flush = 0;



    // Handles VFR by flushing page because this frame needs to have a timestamp

    // For theora, keyframes also need to have a timestamp to correctly mark

    // them as such, otherwise seeking will not work correctly at the very

    // least with old libogg versions.

    // Do not try to flush header packets though, that will create broken files.

    if (st->codec->codec_id == AV_CODEC_ID_THEORA && !header &&

        (ogg_granule_to_timestamp(oggstream, granule) >

         ogg_granule_to_timestamp(oggstream, oggstream->last_granule) + 1 ||

         ogg_key_granule(oggstream, granule))) {

        if (oggstream->page.granule != -1)

            ogg_buffer_page(s, oggstream);

        flush = 1;

    }



    // avoid a continued page

    if (!header && oggstream->page.size > 0 &&

        MAX_PAGE_SIZE - oggstream->page.size < size) {

        ogg_buffer_page(s, oggstream);

    }



    for (i = 0; i < total_segments; ) {

        OGGPage *page = &oggstream->page;



        segments = FFMIN(total_segments - i, 255 - page->segments_count);



        if (i && !page->segments_count)

            page->flags |= 1; // continued packet



        memset(page->segments+page->segments_count, 255, segments - 1);

        page->segments_count += segments - 1;



        len = FFMIN(size, segments*255);

        page->segments[page->segments_count++] = len - (segments-1)*255;

        memcpy(page->data+page->size, p, len);

        p += len;

        size -= len;

        i += segments;

        page->size += len;



        if (i == total_segments)

            page->granule = granule;



        if (!header) {

            AVStream *st = s->streams[page->stream_index];



            int64_t start = av_rescale_q(page->start_granule, st->time_base,

                                         AV_TIME_BASE_Q);

            int64_t next  = av_rescale_q(page->granule, st->time_base,

                                         AV_TIME_BASE_Q);



            if (page->segments_count == 255 ||

                (ogg->pref_size     > 0 && page->size   >= ogg->pref_size) ||

                (ogg->pref_duration > 0 && next - start >= ogg->pref_duration)) {

                ogg_buffer_page(s, oggstream);

            }

        }

    }



    if (flush && oggstream->page.granule != -1)

        ogg_buffer_page(s, oggstream);



    return 0;

}
