static QObject *parse_keyword(JSONParserContext *ctxt)

{

    QObject *token;

    const char *val;



    token = parser_context_pop_token(ctxt);

    assert(token && token_get_type(token) == JSON_KEYWORD);

    val = token_get_value(token);



    if (!strcmp(val, "true")) {

        return QOBJECT(qbool_from_bool(true));

    } else if (!strcmp(val, "false")) {

        return QOBJECT(qbool_from_bool(false));

    } else if (!strcmp(val, "null")) {

        return qnull();

    }

    parse_error(ctxt, token, "invalid keyword '%s'", val);

    return NULL;

}
