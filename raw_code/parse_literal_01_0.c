static QObject *parse_literal(JSONParserContext *ctxt, QList **tokens)

{

    QObject *token, *obj;

    QList *working = qlist_copy(*tokens);



    token = qlist_pop(working);

    if (token == NULL) {

        goto out;

    }



    switch (token_get_type(token)) {

    case JSON_STRING:

        obj = QOBJECT(qstring_from_escaped_str(ctxt, token));

        break;

    case JSON_INTEGER:

        obj = QOBJECT(qint_from_int(strtoll(token_get_value(token), NULL, 10)));

        break;

    case JSON_FLOAT:

        /* FIXME dependent on locale */

        obj = QOBJECT(qfloat_from_double(strtod(token_get_value(token), NULL)));

        break;

    default:

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
