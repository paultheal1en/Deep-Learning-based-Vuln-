char *g_strconcat(const char *s, ...)

{

    char *s;



    /*

     * Can't model: last argument must be null, the others

     * null-terminated strings

     */



    s = __coverity_alloc_nosize__();

    __coverity_writeall__(s);

    __coverity_mark_as_afm_allocated__(s, AFM_free);

    return s;

}
