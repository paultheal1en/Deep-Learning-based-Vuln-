static int srt_encode_frame(AVCodecContext *avctx,

                            unsigned char *buf, int bufsize, const AVSubtitle *sub)

{

    SRTContext *s = avctx->priv_data;

    ASSDialog *dialog;

    int i, len, num;



    s->ptr = s->buffer;

    s->end = s->ptr + sizeof(s->buffer);



    for (i=0; i<sub->num_rects; i++) {



        if (sub->rects[i]->type != SUBTITLE_ASS) {

            av_log(avctx, AV_LOG_ERROR, "Only SUBTITLE_ASS type supported.\n");

            return AVERROR(ENOSYS);

        }



        dialog = ff_ass_split_dialog(s->ass_ctx, sub->rects[i]->ass, 0, &num);

        for (; dialog && num--; dialog++) {

            if (avctx->codec->id == CODEC_ID_SRT) {

                int sh, sm, ss, sc = 10 * dialog->start;

                int eh, em, es, ec = 10 * dialog->end;

                sh = sc/3600000;  sc -= 3600000*sh;

                sm = sc/  60000;  sc -=   60000*sm;

                ss = sc/   1000;  sc -=    1000*ss;

                eh = ec/3600000;  ec -= 3600000*eh;

                em = ec/  60000;  ec -=   60000*em;

                es = ec/   1000;  ec -=    1000*es;

                srt_print(s,"%d\r\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\r\n",

                          ++s->count, sh, sm, ss, sc, eh, em, es, ec);

            }

            s->alignment_applied = 0;

            s->dialog_start = s->ptr - 2;

            srt_style_apply(s, dialog->style);

            ff_ass_split_override_codes(&srt_callbacks, s, dialog->text);

        }

    }



    if (s->ptr == s->buffer)

        return 0;



    len = av_strlcpy(buf, s->buffer, bufsize);



    if (len > bufsize-1) {

        av_log(avctx, AV_LOG_ERROR, "Buffer too small for ASS event.\n");

        return -1;

    }



    return len;

}
