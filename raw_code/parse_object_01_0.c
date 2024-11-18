static QObject *parse_object(JSONParserContext *ctxt, QList **tokens, va_list *ap)

{

    QDict *dict = NULL;

    QObject *token, *peek;

    QList *working = qlist_copy(*tokens);



    token = qlist_pop(working);

    if (token == NULL) {

        goto out;

    }



    if (!token_is_operator(token, '{')) {

        goto out;

    }

    qobject_decref(token);

    token = NULL;



    dict = qdict_new();



    peek = qlist_peek(working);

    if (peek == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    if (!token_is_operator(peek, '}')) {

        if (parse_pair(ctxt, dict, &working, ap) == -1) {

            goto out;

        }



        token = qlist_pop(working);

        if (token == NULL) {

            parse_error(ctxt, NULL, "premature EOI");

            goto out;

        }



        while (!token_is_operator(token, '}')) {

            if (!token_is_operator(token, ',')) {

                parse_error(ctxt, token, "expected separator in dict");

                goto out;

            }

            qobject_decref(token);

            token = NULL;



            if (parse_pair(ctxt, dict, &working, ap) == -1) {

                goto out;

            }



            token = qlist_pop(working);

            if (token == NULL) {

                parse_error(ctxt, NULL, "premature EOI");

                goto out;

            }

        }

        qobject_decref(token);

        token = NULL;

    } else {

        token = qlist_pop(working);

        qobject_decref(token);

        token = NULL;

    }



    QDECREF(*tokens);

    *tokens = working;



    return QOBJECT(dict);



out:

    qobject_decref(token);

    QDECREF(working);

    QDECREF(dict);

    return NULL;

}
