static int has_duration(AVFormatContext *ic)

{

    int i;

    AVStream *st;



    for(i = 0;i < ic->nb_streams; i++) {

        st = ic->streams[i];

        if (st->duration != AV_NOPTS_VALUE)

            return 1;

    }

    if (ic->duration)

        return 1;

    return 0;

}
