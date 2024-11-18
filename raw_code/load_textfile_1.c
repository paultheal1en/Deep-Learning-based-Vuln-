static int load_textfile(AVFilterContext *ctx)

{

    DrawTextContext *s = ctx->priv;

    int err;

    uint8_t *textbuf;

    size_t textbuf_size;



    if ((err = av_file_map(s->textfile, &textbuf, &textbuf_size, 0, ctx)) < 0) {

        av_log(ctx, AV_LOG_ERROR,

               "The text file '%s' could not be read or is empty\n",

               s->textfile);

        return err;

    }



    if (!(s->text = av_realloc(s->text, textbuf_size + 1)))

        return AVERROR(ENOMEM);

    memcpy(s->text, textbuf, textbuf_size);

    s->text[textbuf_size] = 0;

    av_file_unmap(textbuf, textbuf_size);



    return 0;

}
