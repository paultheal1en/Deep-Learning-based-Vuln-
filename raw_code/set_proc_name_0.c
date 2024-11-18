static void set_proc_name(const char *s)

{

#ifdef __linux__

    char name[16];

    if (!s)

        return;

    name[sizeof(name) - 1] = 0;

    strncpy(name, s, sizeof(name));

    /* Could rewrite argv[0] too, but that's a bit more complicated.

       This simple way is enough for `top'. */

    prctl(PR_SET_NAME, name);

#endif    	

}
