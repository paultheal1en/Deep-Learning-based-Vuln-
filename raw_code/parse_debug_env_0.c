int parse_debug_env(const char *name, int max, int initial)

{

    char *debug_env = getenv(name);

    char *inv = NULL;

    int debug;



    if (!debug_env) {

        return initial;

    }

    debug = strtol(debug_env, &inv, 10);

    if (inv == debug_env) {

        return initial;

    }

    if (debug < 0 || debug > max) {

        fprintf(stderr, "warning: %s not in [0, %d]", name, max);

        return initial;

    }

    return debug;

}
