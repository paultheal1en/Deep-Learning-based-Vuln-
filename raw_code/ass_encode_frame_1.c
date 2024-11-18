static int ass_encode_frame(AVCodecContext *avctx,

                            unsigned char *buf, int bufsize,

                            const AVSubtitle *sub)

{

    ASSEncodeContext *s = avctx->priv_data;

    int i, len, total_len = 0;



    for (i=0; i<sub->num_rects; i++) {

        char ass_line[2048];

        const char *ass = sub->rects[i]->ass;



        if (sub->rects[i]->type != SUBTITLE_ASS) {

            av_log(avctx, AV_LOG_ERROR, "Only SUBTITLE_ASS type supported.\n");

            return -1;

        }



        if (strncmp(ass, "Dialogue: ", 10)) {

            av_log(avctx, AV_LOG_ERROR, "AVSubtitle rectangle ass \"%s\""

                   " does not look like a SSA markup\n", ass);

            return AVERROR_INVALIDDATA;

        }



        if (avctx->codec->id == AV_CODEC_ID_ASS) {

            long int layer;

            char *p;



            if (i > 0) {

                av_log(avctx, AV_LOG_ERROR, "ASS encoder supports only one "

                       "ASS rectangle field.\n");

                return AVERROR_INVALIDDATA;

            }



            ass += 10; // skip "Dialogue: "

            /* parse Layer field. If it's a Marked field, the content

             * will be "Marked=N" instead of the layer num, so we will

             * have layer=0, which is fine. */

            layer = strtol(ass, &p, 10);

            if (*p) p += strcspn(p, ",") + 1; // skip layer or marked

            if (*p) p += strcspn(p, ",") + 1; // skip start timestamp

            if (*p) p += strcspn(p, ",") + 1; // skip end timestamp

            snprintf(ass_line, sizeof(ass_line), "%d,%ld,%s", ++s->id, layer, p);

            ass_line[strcspn(ass_line, "\r\n")] = 0;

            ass = ass_line;

        }

        len = av_strlcpy(buf+total_len, ass, bufsize-total_len);



        if (len > bufsize-total_len-1) {

            av_log(avctx, AV_LOG_ERROR, "Buffer too small for ASS event.\n");

            return -1;

        }



        total_len += len;

    }



    return total_len;

}
