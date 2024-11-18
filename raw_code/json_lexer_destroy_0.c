void json_lexer_destroy(JSONLexer *lexer)

{

    QDECREF(lexer->token);

}
