query_params_append (struct QueryParams *ps,

               const char *name, const char *value)

{

    if (ps->n >= ps->alloc) {

        ps->p = g_renew(QueryParam, ps->p, ps->alloc * 2);

        ps->alloc *= 2;

    }



    ps->p[ps->n].name = g_strdup(name);

    ps->p[ps->n].value = value ? g_strdup(value) : NULL;

    ps->p[ps->n].ignore = 0;

    ps->n++;



    return 0;

}
