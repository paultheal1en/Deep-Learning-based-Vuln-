void hmp_hostfwd_add(Monitor *mon, const QDict *qdict)

{

    const char *redir_str;

    SlirpState *s;

    const char *arg1 = qdict_get_str(qdict, "arg1");

    const char *arg2 = qdict_get_try_str(qdict, "arg2");

    const char *arg3 = qdict_get_try_str(qdict, "arg3");



    if (arg2) {

        s = slirp_lookup(mon, arg1, arg2);

        redir_str = arg3;

    } else {

        s = slirp_lookup(mon, NULL, NULL);

        redir_str = arg1;

    }

    if (s) {

        slirp_hostfwd(s, redir_str, 0);

    }



}
