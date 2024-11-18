static void skip_input(DBEContext *s, int nb_words)

{

    s->input      += nb_words * s->word_bytes;

    s->input_size -= nb_words;

}
