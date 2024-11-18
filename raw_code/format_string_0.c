static void format_string(StringOutputVisitor *sov, Range *r, bool next,

                          bool human)

{

    if (r->end - r->begin > 1) {

        if (human) {

            g_string_append_printf(sov->string, "0x%" PRIx64 "-0x%" PRIx64,

                                   r->begin, r->end - 1);



        } else {

            g_string_append_printf(sov->string, "%" PRId64 "-%" PRId64,

                                   r->begin, r->end - 1);

        }

    } else {

        if (human) {

            g_string_append_printf(sov->string, "0x%" PRIx64, r->begin);

        } else {

            g_string_append_printf(sov->string, "%" PRId64, r->begin);

        }

    }

    if (next) {

        g_string_append(sov->string, ",");

    }

}
