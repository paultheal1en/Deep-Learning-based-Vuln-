void error_set(Error **errp, const char *fmt, ...)

{

    Error *err;

    va_list ap;



    if (errp == NULL) {

        return;

    }

    assert(*errp == NULL);



    err = g_malloc0(sizeof(*err));



    va_start(ap, fmt);

    err->obj = qobject_to_qdict(qobject_from_jsonv(fmt, &ap));

    va_end(ap);

    err->msg = qerror_format(fmt, err->obj);



    *errp = err;

}
