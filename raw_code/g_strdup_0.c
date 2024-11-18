char *g_strdup(const char *s)

{

    char *dup;

    size_t i;



    if (!s) {

        return NULL;

    }



    __coverity_string_null_sink__(s);

    __coverity_string_size_sink__(s);

    dup = __coverity_alloc_nosize__();

    __coverity_mark_as_afm_allocated__(dup, AFM_free);

    for (i = 0; (dup[i] = s[i]); i++) ;

    return dup;

}
