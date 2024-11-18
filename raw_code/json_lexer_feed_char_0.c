static int json_lexer_feed_char(JSONLexer *lexer, char ch, bool flush)

{

    int char_consumed, new_state;



    lexer->x++;

    if (ch == '\n') {

        lexer->x = 0;

        lexer->y++;

    }



    do {

        new_state = json_lexer[lexer->state][(uint8_t)ch];

        char_consumed = !TERMINAL_NEEDED_LOOKAHEAD(lexer->state, new_state);

        if (char_consumed) {

            qstring_append_chr(lexer->token, ch);

        }



        switch (new_state) {

        case JSON_OPERATOR:

        case JSON_ESCAPE:

        case JSON_INTEGER:

        case JSON_FLOAT:

        case JSON_KEYWORD:

        case JSON_STRING:

            lexer->emit(lexer, lexer->token, new_state, lexer->x, lexer->y);

        case JSON_SKIP:

            QDECREF(lexer->token);

            lexer->token = qstring_new();

            new_state = IN_START;

            break;

        case IN_ERROR:

            QDECREF(lexer->token);

            lexer->token = qstring_new();

            new_state = IN_START;

            return -EINVAL;

        default:

            break;

        }

        lexer->state = new_state;

    } while (!char_consumed && !flush);



    /* Do not let a single token grow to an arbitrarily large size,

     * this is a security consideration.

     */

    if (lexer->token->length > MAX_TOKEN_SIZE) {

        lexer->emit(lexer, lexer->token, lexer->state, lexer->x, lexer->y);

        QDECREF(lexer->token);

        lexer->token = qstring_new();

        lexer->state = IN_START;

    }



    return 0;

}
