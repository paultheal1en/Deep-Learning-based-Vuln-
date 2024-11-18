static void display_picref(AVFilterBufferRef *picref, AVRational time_base)

{

    int x, y;

    uint8_t *p0, *p;

    int64_t delay;



    if (picref->pts != AV_NOPTS_VALUE) {

        if (last_pts != AV_NOPTS_VALUE) {

            /* sleep roughly the right amount of time;

             * usleep is in microseconds, just like AV_TIME_BASE. */

            delay = av_rescale_q(picref->pts - last_pts,

                                 time_base, AV_TIME_BASE_Q);

            if (delay > 0 && delay < 1000000)

                usleep(delay);

        }

        last_pts = picref->pts;

    }



    /* Trivial ASCII grayscale display. */

    p0 = picref->data[0];

    puts("\033c");

    for (y = 0; y < picref->video->h; y++) {

        p = p0;

        for (x = 0; x < picref->video->w; x++)

            putchar(" .-+#"[*(p++) / 52]);

        putchar('\n');

        p0 += picref->linesize[0];

    }

    fflush(stdout);

}
