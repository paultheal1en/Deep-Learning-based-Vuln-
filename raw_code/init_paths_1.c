void init_paths(const char *prefix)

{

    char pref_buf[PATH_MAX];



    if (prefix[0] == '\0' ||

        !strcmp(prefix, "/"))

        return;



    if (prefix[0] != '/') {

        char *cwd = getcwd(NULL, 0);

        size_t pref_buf_len = sizeof(pref_buf);



        if (!cwd)

            abort();

        pstrcpy(pref_buf, sizeof(pref_buf), cwd);

        pstrcat(pref_buf, pref_buf_len, "/");

        pstrcat(pref_buf, pref_buf_len, prefix);

        free(cwd);

    } else

        pstrcpy(pref_buf, sizeof(pref_buf), prefix + 1);



    base = new_entry("", NULL, pref_buf);

    base = add_dir_maybe(base);

    if (base->num_entries == 0) {

        g_free(base->pathname);

        free(base->name);

        free(base);

        base = NULL;

    } else {

        set_parents(base, base);

    }

}
