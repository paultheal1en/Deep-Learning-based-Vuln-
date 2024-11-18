static void flat_print_str(WriterContext *wctx, const char *key, const char *value)

{

    FlatContext *flat = wctx->priv;

    AVBPrint buf;



    flat_print_key_prefix(wctx);

    av_bprint_init(&buf, 1, AV_BPRINT_SIZE_UNLIMITED);

    printf("%s=", flat_escape_key_str(&buf, key, flat->sep));

    av_bprint_clear(&buf);

    printf("\"%s\"\n", flat_escape_value_str(&buf, value));

    av_bprint_finalize(&buf, NULL);

}
