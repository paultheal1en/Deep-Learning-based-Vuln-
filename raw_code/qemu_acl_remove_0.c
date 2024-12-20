int qemu_acl_remove(qemu_acl *acl,

                    const char *match)

{

    qemu_acl_entry *entry;

    int i = 0;



    TAILQ_FOREACH(entry, &acl->entries, next) {

        i++;

        if (strcmp(entry->match, match) == 0) {

            TAILQ_REMOVE(&acl->entries, entry, next);

            return i;

        }

    }

    return -1;

}
