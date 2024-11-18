static void json_message_process_token(JSONLexer *lexer, GString *input,

                                       JSONTokenType type, int x, int y)

{

    JSONMessageParser *parser = container_of(lexer, JSONMessageParser, lexer);

    JSONToken *token;



    switch (type) {

    case JSON_LCURLY:

        parser->brace_count++;

        break;

    case JSON_RCURLY:

        parser->brace_count--;

        break;

    case JSON_LSQUARE:

        parser->bracket_count++;

        break;

    case JSON_RSQUARE:

        parser->bracket_count--;

        break;

    default:

        break;

    }



    token = g_malloc(sizeof(JSONToken) + input->len + 1);

    token->type = type;

    memcpy(token->str, input->str, input->len);

    token->str[input->len] = 0;

    token->x = x;

    token->y = y;



    parser->token_size += input->len;



    g_queue_push_tail(parser->tokens, token);



    if (type == JSON_ERROR) {

        goto out_emit_bad;

    } else if (parser->brace_count < 0 ||

        parser->bracket_count < 0 ||

        (parser->brace_count == 0 &&

         parser->bracket_count == 0)) {

        goto out_emit;

    } else if (parser->token_size > MAX_TOKEN_SIZE ||

               g_queue_get_length(parser->tokens) > MAX_TOKEN_COUNT ||

               parser->bracket_count + parser->brace_count > MAX_NESTING) {

        /* Security consideration, we limit total memory allocated per object

         * and the maximum recursion depth that a message can force.

         */

        goto out_emit_bad;

    }



    return;



out_emit_bad:

    /*

     * Clear out token list and tell the parser to emit an error

     * indication by passing it a NULL list

     */

    json_message_free_tokens(parser);

out_emit:

    /* send current list of tokens to parser and reset tokenizer */

    parser->brace_count = 0;

    parser->bracket_count = 0;

    /* parser->emit takes ownership of parser->tokens.  */

    parser->emit(parser, parser->tokens);

    parser->tokens = g_queue_new();

    parser->token_size = 0;

}
