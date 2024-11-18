static int find_unused_picture(MpegEncContext *s, int shared)

{

    int i;



    if (shared) {

        for (i = 0; i < MAX_PICTURE_COUNT; i++) {

            if (s->picture[i].f.data[0] == NULL)

                return i;

        }

    } else {

        for (i = 0; i < MAX_PICTURE_COUNT; i++) {

            if (pic_is_unused(s, &s->picture[i]))

                return i;

        }

    }



    av_log(s->avctx, AV_LOG_FATAL,

           "Internal error, picture buffer overflow\n");

    /* We could return -1, but the codec would crash trying to draw into a

     * non-existing frame anyway. This is safer than waiting for a random crash.

     * Also the return of this is never useful, an encoder must only allocate

     * as much as allowed in the specification. This has no relationship to how

     * much libavcodec could allocate (and MAX_PICTURE_COUNT is always large

     * enough for such valid streams).

     * Plus, a decoder has to check stream validity and remove frames if too

     * many reference frames are around. Waiting for "OOM" is not correct at

     * all. Similarly, missing reference frames have to be replaced by

     * interpolated/MC frames, anything else is a bug in the codec ...

     */

    abort();

    return -1;

}
