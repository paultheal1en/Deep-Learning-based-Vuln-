static QObject *parse_escape(JSONParserContext *ctxt, QList **tokens, va_list *ap)

{

    QObject *token = NULL, *obj;

    QList *working = qlist_copy(*tokens);



    if (ap == NULL) {

        goto out;

    }



    token = qlist_pop(working);

    if (token == NULL) {

        goto out;

    }



    if (token_is_escape(token, "%p")) {

        obj = va_arg(*ap, QObject *);

    } else if (token_is_escape(token, "%i")) {

        obj = QOBJECT(qbool_from_int(va_arg(*ap, int)));

    } else if (token_is_escape(token, "%d")) {

        obj = QOBJECT(qint_from_int(va_arg(*ap, int)));

    } else if (token_is_escape(token, "%ld")) {

        obj = QOBJECT(qint_from_int(va_arg(*ap, long)));

    } else if (token_is_escape(token, "%lld") ||

               token_is_escape(token, "%I64d")) {

        obj = QOBJECT(qint_from_int(va_arg(*ap, long long)));

    } else if (token_is_escape(token, "%s")) {

        obj = QOBJECT(qstring_from_str(va_arg(*ap, const char *)));

    } else if (token_is_escape(token, "%f")) {

        obj = QOBJECT(qfloat_from_double(va_arg(*ap, double)));

    } else {

        goto out;

    }



    qobject_decref(token);

    QDECREF(*tokens);

    *tokens = working;



    return obj;



out:

    qobject_decref(token);

    QDECREF(working);



    return NULL;

}
