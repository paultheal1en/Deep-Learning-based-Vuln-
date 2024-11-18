static void add_completion(const char *str)

{

    if (nb_completions < NB_COMPLETIONS_MAX) {

        completions[nb_completions++] = qemu_strdup(str);

    }

}
