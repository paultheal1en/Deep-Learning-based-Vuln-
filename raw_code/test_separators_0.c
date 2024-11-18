static void test_separators(const AVDictionary *m, const char pair, const char val)

{

    AVDictionary *dict = NULL;

    char pairs[] = {pair , '\0'};

    char vals[]  = {val, '\0'};



    char *buffer = NULL;

    av_dict_copy(&dict, m, 0);

    print_dict(dict);

    av_dict_get_string(dict, &buffer, val, pair);

    printf("%s\n", buffer);

    av_dict_free(&dict);

    av_dict_parse_string(&dict, buffer, vals, pairs, 0);

    av_freep(&buffer);

    print_dict(dict);

    av_dict_free(&dict);

}
