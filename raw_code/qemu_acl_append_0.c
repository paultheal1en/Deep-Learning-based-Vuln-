int qemu_acl_append(qemu_acl *acl,

                    int deny,

                    const char *match)

{

    qemu_acl_entry *entry;



    entry = qemu_malloc(sizeof(*entry));

    entry->match = qemu_strdup(match);

    entry->deny = deny;



    TAILQ_INSERT_TAIL(&acl->entries, entry, next);

    acl->nentries++;



    return acl->nentries;

}
