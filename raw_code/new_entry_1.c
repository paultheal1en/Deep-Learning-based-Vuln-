static struct pathelem *new_entry(const char *root,

                                  struct pathelem *parent,

                                  const char *name)

{

    struct pathelem *new = malloc(sizeof(*new));

    new->name = strdup(name);

    new->pathname = g_strdup_printf("%s/%s", root, name);

    new->num_entries = 0;

    return new;

}
