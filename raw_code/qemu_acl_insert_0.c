int qemu_acl_insert(qemu_acl *acl,

                    int deny,

                    const char *match,

                    int index)

{

    qemu_acl_entry *entry;

    qemu_acl_entry *tmp;

    int i = 0;



    if (index <= 0)

        return -1;

    if (index >= acl->nentries)

        return qemu_acl_append(acl, deny, match);





    entry = qemu_malloc(sizeof(*entry));

    entry->match = qemu_strdup(match);

    entry->deny = deny;



    TAILQ_FOREACH(tmp, &acl->entries, next) {

        i++;

        if (i == index) {

            TAILQ_INSERT_BEFORE(tmp, entry, next);

            acl->nentries++;

            break;

        }

    }



    return i;

}
