static struct pathelem *add_entry(struct pathelem *root, const char *name,

                                  unsigned type)

{

    struct pathelem **e;



    root->num_entries++;



    root = realloc(root, sizeof(*root)

                   + sizeof(root->entries[0])*root->num_entries);

    e = &root->entries[root->num_entries-1];



    *e = new_entry(root->pathname, root, name);

    if (is_dir_maybe(type)) {

        *e = add_dir_maybe(*e);

    }



    return root;

}
