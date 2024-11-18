static int mov_open_dref(MOVContext *c, AVIOContext **pb, const char *src, MOVDref *ref,

                         AVIOInterruptCB *int_cb)

{

    AVOpenCallback open_func = c->fc->open_cb;



    if (!open_func)

        open_func = ffio_open2_wrapper;



    /* try relative path, we do not try the absolute because it can leak information about our

       system to an attacker */

    if (ref->nlvl_to > 0 && ref->nlvl_from > 0 && ref->path[0] != '/') {

        char filename[1025];

        const char *src_path;

        int i, l;



        /* find a source dir */

        src_path = strrchr(src, '/');

        if (src_path)

            src_path++;

        else

            src_path = src;



        /* find a next level down to target */

        for (i = 0, l = strlen(ref->path) - 1; l >= 0; l--)

            if (ref->path[l] == '/') {

                if (i == ref->nlvl_to - 1)

                    break;

                else

                    i++;

            }



        /* compose filename if next level down to target was found */

        if (i == ref->nlvl_to - 1 && src_path - src  < sizeof(filename)) {

            memcpy(filename, src, src_path - src);

            filename[src_path - src] = 0;



            for (i = 1; i < ref->nlvl_from; i++)

                av_strlcat(filename, "../", sizeof(filename));



            av_strlcat(filename, ref->path + l + 1, sizeof(filename));

            if (!c->use_absolute_path && !c->fc->open_cb)

                if(strstr(ref->path + l + 1, "..") || ref->nlvl_from > 1)

                    return AVERROR(ENOENT);



            if (strlen(filename) + 1 == sizeof(filename))

                return AVERROR(ENOENT);

            if (!open_func(c->fc, pb, filename, AVIO_FLAG_READ, int_cb, NULL))

                return 0;

        }

    } else if (c->use_absolute_path) {

        av_log(c->fc, AV_LOG_WARNING, "Using absolute path on user request, "

               "this is a possible security issue\n");

        if (!open_func(c->fc, pb, ref->path, AVIO_FLAG_READ, int_cb, NULL))

            return 0;

    } else if (c->fc->open_cb) {

        if (!open_func(c->fc, pb, ref->path, AVIO_FLAG_READ, int_cb, NULL))

            return 0;

    } else {

        av_log(c->fc, AV_LOG_ERROR,

               "Absolute path %s not tried for security reasons, "

               "set demuxer option use_absolute_path to allow absolute paths\n",

               ref->path);

    }



    return AVERROR(ENOENT);

}
