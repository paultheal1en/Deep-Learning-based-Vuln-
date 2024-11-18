static QObject *parse_array(JSONParserContext *ctxt, QList **tokens, va_list *ap)

{

    QList *list = NULL;

    QObject *token, *peek;

    QList *working = qlist_copy(*tokens);



    token = qlist_pop(working);

    if (token == NULL) {

        goto out;

    }



    if (!token_is_operator(token, '[')) {

        goto out;

    }

    qobject_decref(token);

    token = NULL;



    list = qlist_new();



    peek = qlist_peek(working);

    if (peek == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    if (!token_is_operator(peek, ']')) {

        QObject *obj;



        obj = parse_value(ctxt, &working, ap);

        if (obj == NULL) {

            parse_error(ctxt, token, "expecting value");

            goto out;

        }



        qlist_append_obj(list, obj);



        token = qlist_pop(working);

        if (token == NULL) {

            parse_error(ctxt, NULL, "premature EOI");

            goto out;

        }



        while (!token_is_operator(token, ']')) {

            if (!token_is_operator(token, ',')) {

                parse_error(ctxt, token, "expected separator in list");

                goto out;

            }



            qobject_decref(token);

            token = NULL;



            obj = parse_value(ctxt, &working, ap);

            if (obj == NULL) {

                parse_error(ctxt, token, "expecting value");

                goto out;

            }



            qlist_append_obj(list, obj);



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



    return QOBJECT(list);



out:

    qobject_decref(token);

    QDECREF(working);

    QDECREF(list);

    return NULL;

}
