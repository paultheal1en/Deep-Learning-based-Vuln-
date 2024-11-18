static void print_format_entry(const char *tag,

                               const char *val)

{

    if (!fmt_entries_to_show) {

        if (tag) {

            printf("%s=%s\n", tag, val);

        } else {

            printf("%s\n", val);

        }

    } else if (tag && av_dict_get(fmt_entries_to_show, tag, NULL, 0)) {

        if (nb_fmt_entries_to_show > 1)

            printf("%s=", tag);

        printf("%s\n", val);

    }

}
