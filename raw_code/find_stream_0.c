static AVStream *find_stream(void *log, AVFormatContext *avf, const char *spec)

{

    int i, ret, already = 0, stream_id = -1;

    char type_char, dummy;

    AVStream *found = NULL;

    enum AVMediaType type;



    ret = sscanf(spec, "d%[av]%d%c", &type_char, &stream_id, &dummy);

    if (ret >= 1 && ret <= 2) {

        type = type_char == 'v' ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;

        ret = av_find_best_stream(avf, type, stream_id, -1, NULL, 0);

        if (ret < 0) {

            av_log(log, AV_LOG_ERROR, "No %s stream with index '%d' found\n",

                   av_get_media_type_string(type), stream_id);

            return NULL;

        }

        return avf->streams[ret];

    }

    for (i = 0; i < avf->nb_streams; i++) {

        ret = avformat_match_stream_specifier(avf, avf->streams[i], spec);

        if (ret < 0) {

            av_log(log, AV_LOG_ERROR,

                   "Invalid stream specifier \"%s\"\n", spec);

            return NULL;

        }

        if (!ret)

            continue;

        if (avf->streams[i]->discard != AVDISCARD_ALL) {

            already++;

            continue;

        }

        if (found) {

            av_log(log, AV_LOG_WARNING,

                   "Ambiguous stream specifier \"%s\", using #%d\n", spec, i);

            break;

        }

        found = avf->streams[i];

    }

    if (!found) {

        av_log(log, AV_LOG_WARNING, "Stream specifier \"%s\" %s\n", spec,

               already ? "matched only already used streams" :

                         "did not match any stream");

        return NULL;

    }

    if (found->codec->codec_type != AVMEDIA_TYPE_VIDEO &&

        found->codec->codec_type != AVMEDIA_TYPE_AUDIO) {

        av_log(log, AV_LOG_ERROR, "Stream specifier \"%s\" matched a %s stream,"

               "currently unsupported by libavfilter\n", spec,

               av_get_media_type_string(found->codec->codec_type));

        return NULL;

    }

    return found;

}
