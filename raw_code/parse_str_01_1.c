static void parse_str(StringInputVisitor *siv, Error **errp)

{

    char *str = (char *) siv->string;

    long long start, end;

    Range *cur;

    char *endptr;



    if (siv->ranges) {

        return;

    }



    do {

        errno = 0;

        start = strtoll(str, &endptr, 0);

        if (errno == 0 && endptr > str) {

            if (*endptr == '\0') {

                cur = g_malloc0(sizeof(*cur));

                cur->begin = start;

                cur->end = start + 1;

                siv->ranges = g_list_insert_sorted_merged(siv->ranges, cur,

                                                          range_compare);

                cur = NULL;

                str = NULL;

            } else if (*endptr == '-') {

                str = endptr + 1;

                errno = 0;

                end = strtoll(str, &endptr, 0);

                if (errno == 0 && endptr > str && start <= end &&

                    (start > INT64_MAX - 65536 ||

                     end < start + 65536)) {

                    if (*endptr == '\0') {

                        cur = g_malloc0(sizeof(*cur));

                        cur->begin = start;

                        cur->end = end + 1;

                        siv->ranges =

                            g_list_insert_sorted_merged(siv->ranges,

                                                        cur,

                                                        range_compare);

                        cur = NULL;

                        str = NULL;

                    } else if (*endptr == ',') {

                        str = endptr + 1;

                        cur = g_malloc0(sizeof(*cur));

                        cur->begin = start;

                        cur->end = end + 1;

                        siv->ranges =

                            g_list_insert_sorted_merged(siv->ranges,

                                                        cur,

                                                        range_compare);

                        cur = NULL;

                    } else {

                        goto error;

                    }

                } else {

                    goto error;

                }

            } else if (*endptr == ',') {

                str = endptr + 1;

                cur = g_malloc0(sizeof(*cur));

                cur->begin = start;

                cur->end = start + 1;

                siv->ranges = g_list_insert_sorted_merged(siv->ranges,

                                                          cur,

                                                          range_compare);

                cur = NULL;

            } else {

                goto error;

            }

        } else {

            goto error;

        }

    } while (str);



    return;

error:

    g_list_foreach(siv->ranges, free_range, NULL);

    g_list_free(siv->ranges);

    siv->ranges = NULL;

}
