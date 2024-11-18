void do_migrate_set_downtime(Monitor *mon, const QDict *qdict)

{

    char *ptr;

    double d;

    const char *value = qdict_get_str(qdict, "value");



    d = strtod(value, &ptr);

    if (!strcmp(ptr,"ms")) {

        d *= 1000000;

    } else if (!strcmp(ptr,"us")) {

        d *= 1000;

    } else if (!strcmp(ptr,"ns")) {

    } else {

        /* all else considered to be seconds */

        d *= 1000000000;

    }



    max_downtime = (uint64_t)d;

}
