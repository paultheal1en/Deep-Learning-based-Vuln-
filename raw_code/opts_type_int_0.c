opts_type_int(Visitor *v, int64_t *obj, const char *name, Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);

    const QemuOpt *opt;

    const char *str;

    long long val;

    char *endptr;



    if (ov->list_mode == LM_SIGNED_INTERVAL) {

        *obj = ov->range_next.s;

        return;

    }



    opt = lookup_scalar(ov, name, errp);

    if (!opt) {

        return;

    }

    str = opt->str ? opt->str : "";



    /* we've gotten past lookup_scalar() */

    assert(ov->list_mode == LM_NONE || ov->list_mode == LM_IN_PROGRESS);



    errno = 0;

    val = strtoll(str, &endptr, 0);

    if (errno == 0 && endptr > str && INT64_MIN <= val && val <= INT64_MAX) {

        if (*endptr == '\0') {

            *obj = val;

            processed(ov, name);

            return;

        }

        if (*endptr == '-' && ov->list_mode == LM_IN_PROGRESS) {

            long long val2;



            str = endptr + 1;

            val2 = strtoll(str, &endptr, 0);

            if (errno == 0 && endptr > str && *endptr == '\0' &&

                INT64_MIN <= val2 && val2 <= INT64_MAX && val <= val2) {

                ov->range_next.s = val;

                ov->range_limit.s = val2;

                ov->list_mode = LM_SIGNED_INTERVAL;



                /* as if entering on the top */

                *obj = ov->range_next.s;

                return;

            }

        }

    }

    error_set(errp, QERR_INVALID_PARAMETER_VALUE, opt->name,

              (ov->list_mode == LM_NONE) ? "an int64 value" :

                                           "an int64 value or range");

}
