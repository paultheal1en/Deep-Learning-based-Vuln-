static int parse_fade(struct sbg_parser *p, struct sbg_fade *fr)

{

    struct sbg_fade f;



    if (lex_char(p, '<'))

        f.in = SBG_FADE_SILENCE;

    else if (lex_char(p, '-'))

        f.in = SBG_FADE_SAME;

    else if (lex_char(p, '='))

        f.in = SBG_FADE_ADAPT;

    else

        return 0;

    if (lex_char(p, '>'))

        f.out = SBG_FADE_SILENCE;

    else if (lex_char(p, '-'))

        f.out = SBG_FADE_SAME;

    else if (lex_char(p, '='))

        f.out = SBG_FADE_ADAPT;

    else

        return AVERROR_INVALIDDATA;

    *fr = f;

    return 1;

}
