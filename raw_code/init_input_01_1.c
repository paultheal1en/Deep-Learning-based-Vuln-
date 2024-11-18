static int init_input(AVFormatContext *s, const char *filename, AVDictionary **options)

{

    int ret;

    AVProbeData pd = {filename, NULL, 0};



    if(s->iformat && !strlen(filename))

        return 0;



    if (s->pb) {

        s->flags |= AVFMT_FLAG_CUSTOM_IO;

        if (!s->iformat)

            return av_probe_input_buffer(s->pb, &s->iformat, filename, s, 0, 0);

        else if (s->iformat->flags & AVFMT_NOFILE)

            av_log(s, AV_LOG_WARNING, "Custom AVIOContext makes no sense and "

                                      "will be ignored with AVFMT_NOFILE format.\n");

        return 0;

    }



    if ( (s->iformat && s->iformat->flags & AVFMT_NOFILE) ||

        (!s->iformat && (s->iformat = av_probe_input_format(&pd, 0))))

        return 0;



    if ((ret = avio_open2(&s->pb, filename, AVIO_FLAG_READ,

                          &s->interrupt_callback, options)) < 0)

        return ret;

    if (s->iformat)

        return 0;

    return av_probe_input_buffer(s->pb, &s->iformat, filename, s, 0, 0);

}
