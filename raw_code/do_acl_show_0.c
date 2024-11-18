static void do_acl_show(Monitor *mon, const QDict *qdict)

{

    const char *aclname = qdict_get_str(qdict, "aclname");

    qemu_acl *acl = find_acl(mon, aclname);

    qemu_acl_entry *entry;

    int i = 0;



    if (acl) {

        monitor_printf(mon, "policy: %s\n",

                       acl->defaultDeny ? "deny" : "allow");

        TAILQ_FOREACH(entry, &acl->entries, next) {

            i++;

            monitor_printf(mon, "%d: %s %s\n", i,

                           entry->deny ? "deny" : "allow", entry->match);

        }

    }

}
