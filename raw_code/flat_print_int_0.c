static void flat_print_int(WriterContext *wctx, const char *key, long long int value)

{

    flat_print_key_prefix(wctx);

    printf("%s=%lld\n", key, value);

}
