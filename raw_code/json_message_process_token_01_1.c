static void json_message_process_token(JSONLexer *lexer, QString *token, JSONTokenType type, int x, int y)

{

    JSONMessageParser *parser = container_of(lexer, JSONMessageParser, lexer);

    QDict *dict;



    if (type == JSON_OPERATOR) {

        switch (qstring_get_str(token)[0]) {

        case '{':

            parser->brace_count++;

            break;

        case '}':

            parser->brace_count--;

            break;

        case '[':

            parser->bracket_count++;

            break;

        case ']':

            parser->bracket_count--;

            break;

        default:

            break;

        }

    }



    dict = qdict_new();

    qdict_put(dict, "type", qint_from_int(type));

    QINCREF(token);

    qdict_put(dict, "token", token);

    qdict_put(dict, "x", qint_from_int(x));

    qdict_put(dict, "y", qint_from_int(y));



    parser->token_size += token->length;



    qlist_append(parser->tokens, dict);



    if (type == JSON_ERROR) {

        goto out_emit_bad;

    } else if (parser->brace_count < 0 ||

        parser->bracket_count < 0 ||

        (parser->brace_count == 0 &&

         parser->bracket_count == 0)) {

        goto out_emit;

    } else if (parser->token_size > MAX_TOKEN_SIZE ||

               parser->bracket_count > MAX_NESTING ||

               parser->brace_count > MAX_NESTING) {

        /* Security consideration, we limit total memory allocated per object

         * and the maximum recursion depth that a message can force.

         */

        goto out_emit;

    }



    return;



out_emit_bad:

    /* clear out token list and tell the parser to emit and error

     * indication by passing it a NULL list

     */

    QDECREF(parser->tokens);

    parser->tokens = NULL;

out_emit:

    /* send current list of tokens to parser and reset tokenizer */

    parser->brace_count = 0;

    parser->bracket_count = 0;

    parser->emit(parser, parser->tokens);

    if (parser->tokens) {

        QDECREF(parser->tokens);

    }

    parser->tokens = qlist_new();

    parser->token_size = 0;

}
