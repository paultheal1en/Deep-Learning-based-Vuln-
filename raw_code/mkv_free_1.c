static void mkv_free(MatroskaMuxContext *mkv) {


















    if (mkv->main_seekhead) {

        av_freep(&mkv->main_seekhead->entries);

        av_freep(&mkv->main_seekhead);


    if (mkv->cues) {

        av_freep(&mkv->cues->entries);

        av_freep(&mkv->cues);


    if (mkv->attachments) {

        av_freep(&mkv->attachments->entries);

        av_freep(&mkv->attachments);


    av_freep(&mkv->tracks);

    av_freep(&mkv->stream_durations);

    av_freep(&mkv->stream_duration_offsets);
